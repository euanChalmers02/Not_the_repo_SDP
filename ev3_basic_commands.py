import ev3dev.ev3 as ev3
import time

def run(): 
    # A shows the port number it is connected to
    m = ev3.LargeMotor('outA')
    if not m.connected:
        print("Plug a motor into port A")
    else:
        m.run_timed(speed_sp=300, time_sp=1000)
        m.run_timed(speed_sp=300, time_sp=1000)
        # A shows the port number it is connected to
        us = ev3.UltrasonicSensor('in1')
        us.mode = 'US-DIST-CM'
        time.sleep(1)
        print(us.value(), "mm")
        time.sleep(1)
        print(us.value(), "mm")


def motor():
    m = ev3.LargeMotor('outA')
    m.run_timed(speed_sp=300, time_sp=1000)


print('hellow and welcome')
run()
    






