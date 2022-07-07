from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
import os.path as path


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            x, y, z = [int(v) for v in self.path.split('/') if v]
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            filepath = path.abspath(f'sandbox/pyramid/{z}/{x:05}x{y:05}.png')
            print(filepath)
            if not path.isfile(filepath):
                filepath = path.abspath('samples/empty_tile.png')
            with open(filepath, 'rb') as file:
                self.wfile.write(file.read())

        except Exception as e:
            print("error" + str(e))


def main():
    # print('http://127.0.0.1:8080/{x}/{y}/{z}')
    print('http://127.0.0.1:8080/1/1/0')
    httpd = HTTPServer(('localhost', 8080), SimpleHTTPRequestHandler)
    httpd.serve_forever()


if __name__ == '__main__':
    main()
