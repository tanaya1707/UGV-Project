# Electronics
### Code, Architecture & Setup

Component Sheet:
https://docs.google.com/spreadsheets/d/1j_mmpR4PtXuIVBiCf4himiMNayCFcLlo_eKM-ojpYJg/edit#gid=0

## 1) Linear Actuation
### Setup:
1. Upload Code on the Raspberry Pi 4
2. Make all the connections excluding the motor. Cross check the power connection, there is no reverse voltage protection. Do not turn on power yet
3. Before Powering on, Connect the capacitor as close to the DRV-8825 as possible. This protects against voltage spikes
4. Current Rating of the Stepper Motor = 0.31 A/Phase * 2-Phase = 0.62A, For the DRV-8825, Vref = Current_Rating/2 = 0.31V, so set the potential between Vref-via and Vref-potentiometer to 0.31V using a multimeter
5. Check which pairs of wires of the motor are one coil. Do this using an LED, The LED blinks if connected between 2 wires of the same coil when the shaft is rotated
6. Connect the motor and run the code.
