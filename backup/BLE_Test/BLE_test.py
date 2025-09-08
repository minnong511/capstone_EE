# pip install bleak
import asyncio, time
from bleak import BleakClient, BleakScanner

SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
CHAR_UUID    = "12345678-1234-5678-1234-56789abcdef1"
TARGET_NAMES = ["Sensor-01", "Sensor-02", "Sensor-03"]

WINDOW_SEC      = 2.0
SCAN_ROUND_SEC  = 6.0
MAX_SCAN_ROUNDS = 10

def dev_name(d):
    # bleak 버전에 따라 metadata가 없을 수 있으므로 name만 안전하게 사용
    return getattr(d, "name", None)

async def scan_until_at_least_one():
    found = {}
    for round_i in range(1, MAX_SCAN_ROUNDS + 1):
        print(f"[스캔] 라운드 {round_i}/{MAX_SCAN_ROUNDS} ...")
        devices = await BleakScanner.discover(timeout=SCAN_ROUND_SEC)
        for d in devices:
            name = dev_name(d)
            if name in TARGET_NAMES and name not in found:
                found[name] = d
                print(f"  - 발견: {name} ({getattr(d, 'address', 'no-addr')})")
        if found:
            break
    return found

async def main():
    found = await scan_until_at_least_one()
    if not found:
        print("아무 장치도 못 찾음. 광고 중인지/다른 기기에 연결 중인지 확인하세요.")
        return

    targets = list(found.keys())
    print("연결 대상:", targets)

    # Windows에서는 device 객체 자체를 넘기는 것이 종종 더 안정적
    clients = {name: BleakClient(found[name]) for name in targets}
    last_seen = {name: 0.0 for name in targets}

    def make_cb(name):
        def cb(_, data: bytearray):
            ts = time.time()
            last_seen[name] = ts
            try:
                msg = data.decode(errors="ignore")
            except:
                msg = repr(data)
            print(f"[{name}] {msg} @ {ts:.3f}")
        return cb

    print("연결 시도...")
    results = await asyncio.gather(
        *[clients[n].connect(timeout=10.0) for n in targets],
        return_exceptions=True
    )
    connected = []
    for n, r in zip(targets, results):
        if isinstance(r, Exception) or not clients[n].is_connected:
            print(f"  ❌ 연결 실패: {n} -> {r}")
        else:
            print(f"  ✅ 연결 성공: {n}")
            connected.append(n)

    if not connected:
        print("연결된 장치가 없습니다.")
        return

    print("Notify 구독 시작:", connected)
    for n in connected:
        try:
            await clients[n].start_notify(CHAR_UUID, make_cb(n))
        except Exception as e:
            print(f"  ❌ Notify 구독 실패: {n} -> {e}")

    print("하트비트 수신 대기 (CTRL+C로 종료)")
    try:
        while True:
            await asyncio.sleep(0.5)
            now = time.time()
            alive = [n for n, t in last_seen.items() if n in connected and now - t <= WINDOW_SEC]
            if alive:
                print(f"활성 장치: {alive}")
    except KeyboardInterrupt:
        print("종료 중...")
    finally:
        for n in connected:
            try: await clients[n].stop_notify(CHAR_UUID)
            except: pass
            try: await clients[n].disconnect()
            except: pass

if __name__ == "__main__":
    asyncio.run(main())
