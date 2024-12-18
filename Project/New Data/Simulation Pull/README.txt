Hi there!

This folder is where I'll store the files needed for my scripts.
Same Idea as last time, its what I wanted to do, but a lot simpler.

I based my simulation off Program 4.1, Earth.py, modified for Python 3.x
I already had a version of this, and my expansion on it. 

For now Its expanded to only include Jupiter as well, and operates
under the equations of motion for three body systems, equations
4.41 and 4.42 in cpms-ch04 by Professor Wang, but is set up 
to loop for the amount of bodies and add those in the equation
for a sim beyond three bodies (it is unreasonable to add all bodies for 
this project, but fun to look at). All the bodies are saved in 
Planet_Data.py, and pasting them into the sim will add it in. 
The vector data can be modified, and modifying the y-velocity
value for each body will make it look as though it is travelling around
the galaxy, somewhat simulating that enormous orbit as well!
Though unreasonable for what needs to be done, it's still fun to look at.

Gonna save the vector data here, and plug that data into a 
Physics Informed Neural Network (PINN), or a Neural ODE
(they are different, NODEs are 'lighter') that "understands" the 
physical interactions of the objects. The plan right now is to have that 
replace the leapfrog ODE solver (had to make, module was outdated) in my 
sim, and have the predictive ML sim predict the solar system. 

Overall same intention, less messy. 

This was written Friday. Tuesday, technically wednesday, I can say this was a success.
Definitely need tweeks to get better looking visuals, but the attempt at orbit is there. 