# app.py - Complete Flask RAG-Anything API
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import logging
import hashlib
import json
import time
import psutil
import shutil
import schedule
import threading
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from functools import wraps
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import tempfile
import gc

# RAG-Anything imports
try:
    from raganything import RAGAnything
    RAG_AVAILABLE = True
except ImportError:
    print("Warning: raganything not installed. Install with: pip install raganything")
    RAGAnything = None
    RAG_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
class Config:
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    UPLOAD_FOLDER = 'uploads'
    RAG_DATA_FOLDER = 'rag_data'
    ALLOWED_EXTENSIONS = {
        'pdf', 'doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx',
        'txt', 'md', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'
    }
    
    # Environment variables
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    API_KEY = os.getenv('API_KEY')  # For admin endpoints
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    
app.config.from_object(Config)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
app.secret_key = Config.SECRET_KEY

# Use ProxyFix for production deployments behind reverse proxy
if Config.FLASK_ENV == 'production':
    app.wsgi_app = ProxyFix(app.wsgi_app)

# Setup logging
logging.basicConfig(
    level=logging.INFO if Config.FLASK_ENV == 'production' else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.RAG_DATA_FOLDER, exist_ok=True)

# Global storage
rag_instances = {}
start_time = time.time()

# ================================
# UTILITY CLASSES AND FUNCTIONS
# ================================

class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
    
    def is_allowed(self, identifier, limit=100, window=3600):
        with self.lock:
            now = time.time()
            # Clean old requests
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if now - req_time < window
            ]
            
            # Check if under limit
            if len(self.requests[identifier]) < limit:
                self.requests[identifier].append(now)
                return True
            return False

@dataclass
class Metric:
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = None

class MetricsCollector:
    def __init__(self, max_history=1000):
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.counters = defaultdict(int)
        self.lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        with self.lock:
            metric = Metric(name, value, datetime.now(), tags or {})
            self.metrics[name].append(metric)
    
    def increment_counter(self, name: str, tags: Dict[str, str] = None):
        with self.lock:
            key = f"{name}:{json.dumps(tags, sort_keys=True)}" if tags else name
            self.counters[key] += 1
    
    def get_summary(self, hours: int = 1) -> Dict:
        since = datetime.now() - timedelta(hours=hours)
        summary = {}
        
        with self.lock:
            for name, metric_list in self.metrics.items():
                recent_metrics = [m for m in metric_list if m.timestamp >= since]
                if recent_metrics:
                    values = [m.value for m in recent_metrics]
                    summary[name] = {
                        'count': len(values),
                        'avg': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'latest': values[-1]
                    }
            
            summary['counters'] = dict(self.counters)
            
        return summary

# Global instances
rate_limiter = RateLimiter()
metrics = MetricsCollector()

# ================================
# DECORATOR FUNCTIONS
# ================================

def rate_limit(limit=100, window=3600):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            identifier = request.remote_addr
            
            if not rate_limiter.is_allowed(identifier, limit, window):
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'limit': limit,
                    'window': window,
                    'retry_after': window
                }), 429
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def log_request():
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            start_time = time.time()
            
            logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")
            
            try:
                result = f(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record metrics
                status_code = result[1] if isinstance(result, tuple) else 200
                monitor_request(
                    endpoint=request.endpoint or 'unknown',
                    method=request.method,
                    status_code=status_code,
                    duration=duration
                )
                
                logger.info(f"Request completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                monitor_request(
                    endpoint=request.endpoint or 'unknown',
                    method=request.method,
                    status_code=500,
                    duration=duration
                )
                logger.error(f"Request failed after {duration:.2f}s: {str(e)}")
                raise
                
        return decorated_function
    return decorator

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        expected_key = Config.API_KEY
        
        if expected_key and api_key != expected_key:
            return jsonify({'error': 'Invalid or missing API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

def validate_json():
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# ================================
# UTILITY FUNCTIONS
# ================================

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def get_file_hash(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_system_stats():
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory': {
            'total': psutil.virtual_memory().total,
            'available': psutil.virtual_memory().available,
            'percent': psutil.virtual_memory().percent
        },
        'disk': {
            'total': psutil.disk_usage('/').total,
            'free': psutil.disk_usage('/').free,
            'percent': psutil.disk_usage('/').percent
        }
    }

def monitor_system():
    try:
        metrics.record_metric('cpu_percent', psutil.cpu_percent())
        memory = psutil.virtual_memory()
        metrics.record_metric('memory_percent', memory.percent)
        metrics.record_metric('memory_available', memory.available)
        
        disk = psutil.disk_usage('/')
        metrics.record_metric('disk_percent', disk.percent)
        metrics.record_metric('disk_free', disk.free)
        
        metrics.record_metric('active_rag_instances', len(rag_instances))
    except Exception as e:
        logger.error(f"Error monitoring system: {e}")

def monitor_request(endpoint: str, method: str, status_code: int, duration: float):
    tags = {
        'endpoint': endpoint,
        'method': method,
        'status_code': str(status_code)
    }
    metrics.record_metric('request_duration', duration, tags)
    metrics.increment_counter('requests_total', tags)
    
    if status_code >= 400:
        metrics.increment_counter('requests_errors', tags)

def cleanup_old_files(max_age_hours=24):
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    cleaned_files = 0
    
    try:
        # Clean upload folder
        for filename in os.listdir(Config.UPLOAD_FOLDER):
            file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time < cutoff_time:
                    try:
                        os.remove(file_path)
                        cleaned_files += 1
                    except Exception as e:
                        logger.error(f"Error removing file {file_path}: {e}")
        
        # Clean RAG data folder
        for dirname in os.listdir(Config.RAG_DATA_FOLDER):
            dir_path = os.path.join(Config.RAG_DATA_FOLDER, dirname)
            if os.path.isdir(dir_path):
                dir_time = datetime.fromtimestamp(os.path.getmtime(dir_path))
                if dir_time < cutoff_time:
                    try:
                        shutil.rmtree(dir_path)
                        cleaned_files += 1
                    except Exception as e:
                        logger.error(f"Error removing directory {dir_path}: {e}")
        
        # Clean old RAG instances from memory
        to_remove = []
        for file_id, rag_data in rag_instances.items():
            if rag_data['created_at'] < cutoff_time.replace(tzinfo=None):
                to_remove.append(file_id)
        
        for file_id in to_remove:
            del rag_instances[file_id]
            cleaned_files += 1
        
        logger.info(f"Cleanup completed: {cleaned_files} items removed")
        return cleaned_files
        
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return 0

def initialize_rag_instance(file_hash, file_path):
    if not RAG_AVAILABLE:
        raise ImportError("RAG-Anything not available")
    
    try:
        work_dir = os.path.join(Config.RAG_DATA_FOLDER, file_hash)
        os.makedirs(work_dir, exist_ok=True)
        
        logger.info(f"Initializing RAG for file: {file_path}")
        
        # Initialize RAG-Anything with configuration
        from raganything.config import RAGAnythingConfig
        
        config = RAGAnythingConfig(
            working_dir=work_dir
        )
        
        # Configure lightrag_kwargs for proper LightRAG initialization
        lightrag_kwargs = {
            'working_dir': work_dir,
        }
        
        # Configure LLM model function if OpenAI API key is available
        llm_model_func = None
        embedding_func = None
        if Config.OPENAI_API_KEY:
            from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
            llm_model_func = gpt_4o_mini_complete
            embedding_func = openai_embed
            lightrag_kwargs.update({
                'llm_model_func': llm_model_func,
                'embedding_func': embedding_func,
            })
        
        rag = RAGAnything(
            config=config, 
            lightrag_kwargs=lightrag_kwargs,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func
        )
        
        # Process the document
        logger.info(f"Processing document through RAG-Anything: {file_path}")
        
        # Handle async operations properly
        import asyncio
        import threading
        
        async_error = None
        
        def run_async_in_thread():
            nonlocal async_error
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Ensure LightRAG is initialized first
                loop.run_until_complete(rag._ensure_lightrag_initialized())
                # Then process the document
                loop.run_until_complete(rag.process_document_complete(file_path))
            except Exception as e:
                async_error = e
                logger.error(f"Async processing error: {str(e)}")
            finally:
                loop.close()
        
        # Run async operations in separate thread to avoid event loop conflicts
        thread = threading.Thread(target=run_async_in_thread)
        thread.start()
        thread.join()
        
        # Check if async processing failed
        if async_error:
            raise async_error
        
        # Store the instance
        rag_instances[file_hash] = {
            'rag': rag,
            'created_at': datetime.now(),
            'file_path': file_path,
            'work_dir': work_dir,
            'original_filename': os.path.basename(file_path)
        }
        
        logger.info(f"RAG instance created successfully for file_id: {file_hash}")
        return rag
        
    except Exception as e:
        logger.error(f"Error initializing RAG instance: {str(e)}")
        # Cleanup on failure
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        raise

# ================================
# ERROR HANDLERS
# ================================

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({'error': 'An unexpected error occurred'}), 500

# ================================
# MAIN API ENDPOINTS
# ================================

@app.route('/health', methods=['GET'])
def health_check():
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'rag_available': RAG_AVAILABLE,
            'active_instances': len(rag_instances),
            'uptime_seconds': time.time() - start_time,
            'version': '1.0.0'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/health/detailed', methods=['GET'])
def detailed_health():
    try:
        system_stats = get_system_stats()
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'uptime_seconds': time.time() - start_time,
            'dependencies': {
                'rag_available': RAG_AVAILABLE,
                'openai_api_configured': bool(Config.OPENAI_API_KEY),
            },
            'system': system_stats,
            'application': {
                'active_rag_instances': len(rag_instances),
                'total_requests': sum(metrics.counters.values()),
                'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/upload', methods=['POST'])
@rate_limit(limit=10, window=3600)  # 10 uploads per hour
@log_request()
def upload_file():
    try:
        # Check if RAG-Anything is available
        if not RAG_AVAILABLE:
            return jsonify({
                'error': 'RAG-Anything not available. Please install with: pip install raganything'
            }), 503
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'File type not supported. Allowed types: {", ".join(Config.ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
        
        # Generate file hash for unique identification
        file_hash = get_file_hash(file_path)
        
        # Check if this file was already processed
        if file_hash in rag_instances:
            # Remove the duplicate uploaded file
            os.remove(file_path)
            return jsonify({
                'message': 'File already processed',
                'file_id': file_hash,
                'filename': rag_instances[file_hash]['original_filename'],
                'status': 'ready',
                'processed_at': rag_instances[file_hash]['created_at'].isoformat()
            })
        
        # Process with RAG-Anything
        try:
            rag = initialize_rag_instance(file_hash, file_path)
            
            # Record successful upload
            metrics.increment_counter('uploads_success')
            
            return jsonify({
                'message': 'File uploaded and processed successfully',
                'file_id': file_hash,
                'filename': filename,
                'status': 'ready',
                'processed_at': datetime.now().isoformat(),
                'file_size_bytes': os.path.getsize(file_path)
            })
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            metrics.increment_counter('uploads_failed')
            
            # Cleanup on failure
            if os.path.exists(file_path):
                os.remove(file_path)
                
            return jsonify({
                'error': 'Failed to process file with RAG-Anything',
                'details': str(e)
            }), 500
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        metrics.increment_counter('uploads_failed')
        return jsonify({'error': 'Upload failed', 'details': str(e)}), 500

@app.route('/chat', methods=['POST'])
@rate_limit(limit=100, window=3600)  # 100 chats per hour
@log_request()
@validate_json()
def chat():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        file_id = data.get('file_id')
        query = data.get('query', '').strip()
        mode = data.get('mode', 'hybrid')  # hybrid, naive, local, global
        
        # Validation
        if not file_id:
            return jsonify({'error': 'file_id is required'}), 400
        
        if not query:
            return jsonify({'error': 'query is required and cannot be empty'}), 400
        
        if len(query) > 1000:
            return jsonify({'error': 'Query too long. Maximum 1000 characters allowed'}), 400
        
        # Check if RAG instance exists for this file
        if file_id not in rag_instances:
            return jsonify({
                'error': 'File not found or not processed. Please upload the file first.',
                'file_id': file_id
            }), 404
        
        # Get RAG instance
        rag_data = rag_instances[file_id]
        rag = rag_data['rag']
        
        # Perform query
        try:
            logger.info(f"Processing query for file_id {file_id}: {query[:100]}...")
            
            start_query_time = time.time()
            
            # Query the RAG system (handle async if needed)
            if hasattr(rag.query, '__call__'):
                # Check if it's an async method
                import asyncio
                import inspect
                
                if inspect.iscoroutinefunction(rag.query):
                    # Handle async query
                    def run_query_async():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            return loop.run_until_complete(rag.query(query=query, mode=mode))
                        finally:
                            loop.close()
                    
                    response = run_query_async()
                else:
                    # Regular sync query
                    response = rag.query(query=query, param=mode)
            else:
                response = rag.query(query=query, param=mode)
            
            query_duration = time.time() - start_query_time
            
            # Record metrics
            metrics.record_metric('query_duration', query_duration, {'mode': mode})
            metrics.increment_counter('queries_success', {'mode': mode})
            
            result = {
                'response': response,
                'query': query,
                'mode': mode,
                'file_id': file_id,
                'filename': rag_data['original_filename'],
                'timestamp': datetime.now().isoformat(),
                'processing_time_seconds': round(query_duration, 2)
            }
            
            logger.info(f"Query processed successfully in {query_duration:.2f}s")
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Query error for file_id {file_id}: {str(e)}")
            metrics.increment_counter('queries_failed', {'mode': mode})
            
            return jsonify({
                'error': 'Failed to process query',
                'details': str(e),
                'file_id': file_id,
                'query': query
            }), 500
            
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({'error': 'Chat request failed', 'details': str(e)}), 500

@app.route('/chat/knowledge-base', methods=['POST'])
@rate_limit(limit=100, window=3600)  # 100 chats per hour
@log_request()
@validate_json()
def chat_knowledge_base():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        query = data.get('query', '').strip()
        mode = data.get('mode', 'hybrid')  # hybrid, naive, local, global
        
        # Validation
        if not query:
            return jsonify({'error': 'query is required and cannot be empty'}), 400
        
        if len(query) > 1000:
            return jsonify({'error': 'Query too long. Maximum 1000 characters allowed'}), 400
        
        # Check if there are any RAG instances available
        if not rag_instances:
            return jsonify({
                'error': 'No files in knowledge base. Please upload files first.',
                'available_files': 0
            }), 404
        
        # Query all RAG instances and collect responses
        responses = []
        total_processing_time = 0
        processed_files = []
        failed_files = []
        
        logger.info(f"Processing knowledge base query: {query[:100]}...")
        start_query_time = time.time()
        
        for file_id, rag_data in rag_instances.items():
            try:
                file_start_time = time.time()
                rag = rag_data['rag']
                
                # Query this RAG instance (handle async if needed)
                if hasattr(rag.query, '__call__'):
                    # Check if it's an async method
                    import asyncio
                    import inspect
                    
                    if inspect.iscoroutinefunction(rag.query):
                        # Handle async query
                        def run_query_async():
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                return loop.run_until_complete(rag.query(query=query, mode=mode))
                            finally:
                                loop.close()
                        
                        response = run_query_async()
                    else:
                        # Regular sync query
                        response = rag.query(query=query, param=mode)
                else:
                    response = rag.query(query=query, param=mode)
                
                file_duration = time.time() - file_start_time
                
                responses.append({
                    'file_id': file_id,
                    'filename': rag_data['original_filename'],
                    'response': response,
                    'processing_time_seconds': round(file_duration, 2)
                })
                
                processed_files.append(rag_data['original_filename'])
                total_processing_time += file_duration
                
            except Exception as e:
                logger.error(f"Error querying file {file_id}: {str(e)}")
                failed_files.append({
                    'file_id': file_id,
                    'filename': rag_data['original_filename'],
                    'error': str(e)
                })
        
        total_query_duration = time.time() - start_query_time
        
        # Record metrics
        metrics.record_metric('knowledge_base_query_duration', total_query_duration, {'mode': mode})
        metrics.increment_counter('knowledge_base_queries_success', {'mode': mode, 'files_count': str(len(processed_files))})
        
        if failed_files:
            metrics.increment_counter('knowledge_base_queries_partial_failure', {'mode': mode})
        
        result = {
            'responses': responses,
            'summary': {
                'query': query,
                'mode': mode,
                'total_files_processed': len(processed_files),
                'total_files_failed': len(failed_files),
                'processed_files': processed_files,
                'failed_files': failed_files if failed_files else None,
                'total_processing_time_seconds': round(total_processing_time, 2),
                'total_query_time_seconds': round(total_query_duration, 2)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Knowledge base query processed: {len(processed_files)} files in {total_query_duration:.2f}s")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Knowledge base chat error: {str(e)}")
        metrics.increment_counter('knowledge_base_queries_failed')
        return jsonify({'error': 'Knowledge base query failed', 'details': str(e)}), 500

@app.route('/files', methods=['GET'])
@log_request()
def list_files():
    try:
        files_info = []
        for file_id, rag_data in rag_instances.items():
            file_info = {
                'file_id': file_id,
                'filename': rag_data['original_filename'],
                'file_path': os.path.basename(rag_data['file_path']),
                'created_at': rag_data['created_at'].isoformat(),
                'status': 'ready',
                'work_dir': rag_data['work_dir']
            }
            
            # Add file size if file still exists
            if os.path.exists(rag_data['file_path']):
                file_info['file_size_bytes'] = os.path.getsize(rag_data['file_path'])
            
            files_info.append(file_info)
        
        # Sort by creation time (newest first)
        files_info.sort(key=lambda x: x['created_at'], reverse=True)
        
        return jsonify({
            'files': files_info,
            'count': len(files_info),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"List files error: {str(e)}")
        return jsonify({'error': 'Failed to list files', 'details': str(e)}), 500

@app.route('/files/<file_id>', methods=['DELETE'])
@log_request()
def delete_file(file_id):
    try:
        if file_id not in rag_instances:
            return jsonify({
                'error': 'File not found',
                'file_id': file_id
            }), 404
        
        rag_data = rag_instances[file_id]
        filename = rag_data['original_filename']
        
        # Clean up files
        cleanup_errors = []
        try:
            # Remove uploaded file
            if os.path.exists(rag_data['file_path']):
                os.remove(rag_data['file_path'])
                
        except Exception as e:
            cleanup_errors.append(f"Failed to remove uploaded file: {str(e)}")
        
        try:
            # Remove RAG working directory
            if os.path.exists(rag_data['work_dir']):
                shutil.rmtree(rag_data['work_dir'])
                
        except Exception as e:
            cleanup_errors.append(f"Failed to remove work directory: {str(e)}")
        
        # Remove from memory
        del rag_instances[file_id]
        
        metrics.increment_counter('files_deleted')
        
        result = {
            'message': 'File deleted successfully',
            'file_id': file_id,
            'filename': filename,
            'timestamp': datetime.now().isoformat()
        }
        
        if cleanup_errors:
            result['cleanup_warnings'] = cleanup_errors
            logger.warning(f"Cleanup warnings for {file_id}: {cleanup_errors}")
        
        logger.info(f"File {file_id} ({filename}) deleted successfully")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Delete file error: {str(e)}")
        return jsonify({'error': 'Failed to delete file', 'details': str(e)}), 500

@app.route('/files/<file_id>/info', methods=['GET'])
@log_request()
def get_file_info(file_id):
    try:
        if file_id not in rag_instances:
            return jsonify({
                'error': 'File not found',
                'file_id': file_id
            }), 404
        
        rag_data = rag_instances[file_id]
        
        file_info = {
            'file_id': file_id,
            'filename': rag_data['original_filename'],
            'created_at': rag_data['created_at'].isoformat(),
            'status': 'ready',
            'file_path': rag_data['file_path'],
            'work_dir': rag_data['work_dir']
        }
        
        # Add file stats if available
        if os.path.exists(rag_data['file_path']):
            stat = os.stat(rag_data['file_path'])
            file_info.update({
                'file_size_bytes': stat.st_size,
                'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        return jsonify(file_info)
        
    except Exception as e:
        logger.error(f"Get file info error: {str(e)}")
        return jsonify({'error': 'Failed to get file info', 'details': str(e)}), 500

# ================================
# ADMIN ENDPOINTS
# ================================

@app.route('/admin/stats', methods=['GET'])
@require_api_key
@log_request()
def admin_stats():
    try:
        hours = int(request.args.get('hours', 1))
        
        stats = get_system_stats()
        metrics_summary = metrics.get_summary(hours)
        
        stats.update({
            'application': {
                'active_files': len(rag_instances),
                'uptime_seconds': time.time() - start_time,
                'rag_available': RAG_AVAILABLE,
                'environment': Config.FLASK_ENV
            },
            'metrics': metrics_summary,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Admin stats error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/admin/cleanup', methods=['POST'])
@require_api_key
@log_request()
def admin_cleanup():
    try:
        max_age = request.json.get('max_age_hours', 24) if request.is_json else 24
        max_age = max(1, min(max_age, 168))  # Between 1 hour and 1 week
        
        cleaned = cleanup_old_files(max_age)
        
        # Force garbage collection
        gc.collect()
        
        return jsonify({
            'message': f'Cleaned {cleaned} old files/directories',
            'max_age_hours': max_age,
            'timestamp': datetime.now().isoformat(),
            'remaining_files': len(rag_instances)
        })
    except Exception as e:
        logger.error(f"Admin cleanup error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/admin/memory', methods=['GET'])
@require_api_key
@log_request()
def admin_memory():
    try:
        import sys
        
        # Force garbage collection
        collected = gc.collect()
        
        memory_info = {
            'rag_instances': {
                'count': len(rag_instances),
                'details': {
                    file_id: {
                        'filename': data['original_filename'],
                        'created_at': data['created_at'].isoformat(),
                        'work_dir_size': get_dir_size(data['work_dir']) if os.path.exists(data['work_dir']) else 0
                    }
                    for file_id, data in rag_instances.items()
                }
            },
            'system_memory': get_system_stats()['memory'],
            'process_memory': {
                'rss_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                'vms_mb': psutil.Process().memory_info().vms / 1024 / 1024,
                'percent': psutil.Process().memory_percent()
            },
            'garbage_collection': {
                'collected_objects': collected,
                'garbage_count': len(gc.garbage),
                'stats': gc.get_stats()
            },
            'python_info': {
                'version': sys.version,
                'platform': sys.platform
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(memory_info)
    except Exception as e:
        logger.error(f"Admin memory error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def get_dir_size(path):
    """Calculate directory size in bytes"""
    try:
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)
        return total
    except:
        return 0

@app.route('/metrics', methods=['GET'])
def metrics_endpoint():
    try:
        hours = int(request.args.get('hours', 1))
        hours = max(1, min(hours, 24))  # Between 1 and 24 hours
        
        return jsonify(metrics.get_summary(hours))
    except Exception as e:
        logger.error(f"Metrics endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ================================
# BACKGROUND TASKS
# ================================

def background_scheduler():
    """Background thread for scheduled tasks"""
    try:
        # Schedule periodic tasks
        schedule.every(6).hours.do(lambda: cleanup_old_files(24))  # Cleanup every 6 hours
        schedule.every(1).hours.do(monitor_system)  # Monitor system every hour
        schedule.every(30).minutes.do(lambda: gc.collect())  # Garbage collection every 30 mins
        
        logger.info("Background scheduler started")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    except Exception as e:
        logger.error(f"Background scheduler error: {e}")

# Start background scheduler thread
scheduler_thread = threading.Thread(target=background_scheduler, daemon=True)
scheduler_thread.start()

# ================================
# APPLICATION STARTUP
# ================================

def check_dependencies():
    """Check if all required dependencies are available"""
    issues = []
    
    if not RAG_AVAILABLE:
        issues.append("RAG-Anything not installed. Install with: pip install raganything")
    
    if not Config.OPENAI_API_KEY:
        issues.append("OPENAI_API_KEY not set. Set environment variable for LLM functionality")
    
    if not os.path.exists(Config.UPLOAD_FOLDER):
        try:
            os.makedirs(Config.UPLOAD_FOLDER)
        except Exception as e:
            issues.append(f"Cannot create upload folder: {e}")
    
    if not os.path.exists(Config.RAG_DATA_FOLDER):
        try:
            os.makedirs(Config.RAG_DATA_FOLDER)
        except Exception as e:
            issues.append(f"Cannot create RAG data folder: {e}")
    
    return issues

# ================================
# MAIN APPLICATION
# ================================

if __name__ == '__main__':
    # Check dependencies at startup
    issues = check_dependencies()
    if issues:
        print("⚠️  Startup warnings:")
        for issue in issues:
            print(f"   - {issue}")
        print()
    
    # Print startup information
    print("🚀 RAG-Anything Flask API Starting...")
    print(f"   Environment: {Config.FLASK_ENV}")
    print(f"   RAG-Anything Available: {RAG_AVAILABLE}")
    print(f"   OpenAI API Configured: {bool(Config.OPENAI_API_KEY)}")
    print(f"   Upload Folder: {Config.UPLOAD_FOLDER}")
    print(f"   RAG Data Folder: {Config.RAG_DATA_FOLDER}")
    print()
    
    # Development vs Production settings
    if Config.FLASK_ENV == 'development':
        print("🔧 Running in DEVELOPMENT mode")
        app.run(
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 5005)),
            debug=True,
            use_reloader=False  # Disable reloader to prevent issues with background threads
        )
    else:
        print("🏭 Running in PRODUCTION mode")
        print("   Note: Use Gunicorn or similar WSGI server for production deployment")
        print("   Example: gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 300 app:app")
        app.run(
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 5005)),
            debug=False
        )