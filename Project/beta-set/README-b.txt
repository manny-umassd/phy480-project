This beta file was made to try a different approach. 

Instead of feeding the data sets directly into the GNN
the goal here is to merge the two sets to create a single
set of trajectories, in hopes that the model would be 
able to recognize the shape of the graph and produce some-
thing versus the blank graph in the alpha set's results.

This approach was a success, but we see that the model 
had merged all orbital data into one impossible celestial
object. 

More work needs to be done to ensure the model can clearly
distinguish the different planets to predict the full set
of objects in the system effectively. 

I could probably pull this off by editing the data pre-
processing script and trying once more to merge by planet, 
or the issue is in the model and how it interprets the new
dataset. 