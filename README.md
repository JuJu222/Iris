# Iris

Artificial Intelligence Final Project

Justin Jap - 0706012010042 <br/>
Kevin Sander Utomo - 0706012010064 <br/>
Kenny Jinhiro Wibowo - 0706012010024 <br/>

###### What is Iris?
Iris is an AI system designed with the purpose of preventing Digital Eye Strain (DES) when using Visual Display Terminals (VDTs) for extensive periods of time by warning the user if they are not blinking in the normal range of blinks per minute.

###### Instructions
1. Extract the zip file of the application that has been downloaded <br />
2. Make sure that the extracted files are in the same folder <br />
3. To launch the app, open the terminal or any CLI and run the command
> python Iris.py {Webcam Number (0 for default webcam)} <br /><br />Example: <br />python Iris.py 0 <br />
4. A window with your webcam feed will show up and the app is ready to detect your face <br />
6. A blue rectangle box will show up around your face if it succesfully detects you
7. Green rectangle boxes will show up around your eyes if it detects that your eyes are open
8. Red rectangle boxes will show up around your eyes if it detects that your eyes are closed
9. Iris will warn you if your number of blinks per minute is below the optimal amount (17 blinks or above per minute) 
10. Press the escape key (ESC) on the keyboard to stop the application from running <br />
