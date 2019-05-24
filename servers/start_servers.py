import subprocess

call_charlie = ["python", "run_websocket_server.py", "--port", "8776", "--id", "charlie"]

call_alice = ["python", "run_websocket_server.py", "--port", "8777", "--id", "alice"]

call_bob = ["python", "run_websocket_server.py", "--port", "8778", "--id", "bob"]

call_crypto = ["python", "run_websocket_server.py", "--port", "8779", "--id", "cypto_provider"]

print("Starting server for Alice")
subprocess.Popen(call_alice)

print("Starting server for Bob")
subprocess.Popen(call_bob)

print("Starting server for Charlie")
subprocess.Popen(call_charlie)

print("Starting server for Crypto provider")
subprocess.Popen(call_crypto)
