import websocket
import json
import rel

def on_message(ws, message):
    print(message)

def on_error(ws, error):
    print(error)

def on_close(ws, close_status_code, close_msg):
    print("### closed ###")

def on_open(ws):
    print("Opened connection")
    logon_request = {
        "Type": 1,
        "ProtocolVersion": 8,
        "Username": "manisd",
        "Password": "NFTZUWy6btg74!@",
        "HeartbeatIntervalInSeconds": 30,
        "Integer1": 0x4 | 0x80 | 0x800 | 0x80000 | 0x100000
    }
    ws.send(json.dumps(logon_request, ensure_ascii=True).encode('latin-1') + b'\x00')
    # Response
    logon_response = ws.recv()
    print("Logon Response:", logon_response)

    if '"Result":1' not in logon_response:
        print("Logon Failed!")
        ws.close()
        exit()

if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("ws://localhost:11099",
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)

    ws.run_forever(dispatcher=rel, reconnect=5)  # Set dispatcher to automatic reconnection, 5 second reconnect delay if connection closed unexpectedly
    rel.signal(2, rel.abort)  # Keyboard Interrupt
    rel.dispatch()