import RPi.GPIO as GPIO
import time


class MotorControl:

    motors = [[17, 18, 27, 22], [23, 24, 25, 4]]
    step_sequence = [[1, 0, 0, 1], [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]]

    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        for motor in self.motors:
            for pin in motor:
                GPIO.setup(pin, GPIO.OUT)

    def setStep(self, sequence):
        for motor in self.motors:
            for i in range(len(motor)):
                GPIO.output(motor[i], sequence[i])

    def forward(self, delay, steps):
        for i in range(0, steps):
            self.setStep(self.step_sequence[i % 8])
            time.sleep(delay)

    def backward(self, delay, steps):
        for i in range(steps-1, 0, -1):
            self.setStep(self.step_sequence[i % 8])
            time.sleep(delay)

    def cleanUp(self):
        GPIO.cleanup()


#gpio pins for motor 1
#coil_A_1_pin = 17
#coil_A_2_pin = 18
#coil_B_1_pin = 27
#coil_B_2_pin = 22

#gpio pins for motor 2
#coil_A_1_pin = 23
#coil_A_2_pin = 24
#coil_B_1_pin = 25
#coil_B_2_pin = 4
