from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SignSpeak ASL Recognition</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .warning { background: #fff3cd; padding: 20px; border-radius: 8px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>SignSpeak ASL Recognition</h1>
                <div class="warning">
                    <h3>⚠️ Vercel Limitation Notice</h3>
                    <p>This ASL recognition application requires:</p>
                    <ul>
                        <li>Real-time camera access</li>
                        <li>Machine learning libraries (TensorFlow, PyTorch)</li>
                        <li>Continuous processing capabilities</li>
                    </ul>
                    <p><strong>Vercel's serverless platform cannot support these features.</strong></p>
                </div>
                
                <h3>🎯 Recommended Hosting Options:</h3>
                <ul>
                    <li><strong>Streamlit Community Cloud</strong> - Free, perfect for Streamlit apps</li>
                    <li><strong>Render.com</strong> - Great for ML applications</li>
                    <li><strong>Railway.app</strong> - Easy deployment with Docker support</li>
                    <li><strong>Hugging Face Spaces</strong> - Specifically for ML demos</li>
                </ul>
                
                <p>For full functionality, please deploy to one of the recommended platforms.</p>
            </div>
        </body>
        </html>
        """
        
        self.wfile.write(html.encode())
        return