import cv2

from windows_iphone_client_v2 import IPhoneRemote


# Replace with the IP address of your Mac.
MAC_IP = "10.10.10.1"


iphone = IPhoneRemote(
    mac_ip=MAC_IP,
)

frame = iphone.get_screen(
    wait_new=False,
    timeout=10,
)

print("First frame:", frame.shape)


def mouse(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Never wait for XCTest inside the OpenCV callback. The callback runs
        # on the same thread as waitKey()/imshow, so a synchronous send_tap()
        # makes the video window appear frozen for the RPC duration.
        future = iphone.send_tap_async(x, y)
        print("tap queued:", x, y)

        def done(f):
            try:
                f.result()
            except Exception as exc:
                print("tap failed:", exc)

        future.add_done_callback(done)


cv2.namedWindow("iPhone", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("iPhone", mouse)

try:
    while True:
        try:
            frame = iphone.get_screen(
                wait_new=True,
                timeout=1.0,
                copy=False,
            )
        except TimeoutError:
            continue

        cv2.imshow("iPhone", frame)

        if (cv2.waitKey(1) & 0xFF) == 27:
            break

finally:
    iphone.close()
    cv2.destroyAllWindows()
