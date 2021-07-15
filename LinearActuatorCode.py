import time
import RPi.GPIO as GPIO

direction = 20
step = 21

GPIO.setmode(GPIO.BCM)
GPIO.setup(direction, GPIO.OUT)
GPIO.setup(step, GPIO.OUT)

def linear_actuate(current_height, desired_height):
    
    one_step_time = 0.208
    speed = 10 #cm/step
    steps_required = abs(round((desired_height - current_height)/speed))

    cw = 1
    ccw = 0

    if desired_height - current_height > 0:
        GPIO.output(direction, cw)
    else:
        GPIO.output(direction, ccw)


    for i in range(steps_required):
        GPIO.output(step, GPIO.HIGH)
        time.sleep(one_step_time)
    GPIO.output(step, GPIO.LOW)



