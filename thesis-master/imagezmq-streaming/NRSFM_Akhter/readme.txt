The datasets and results for our experiments are given in the mat files
uploaded on http://cvlab.lums.edu.pk/nrsfm. Out of these Stretch, Yoga,
Pickup, Drink, Dance, Shark are synthetic datasets, while Dinosaur, 
cubes and Matrix are real datasets. Following are the contents of these 
mat files,

Synthetic Datasets:
Measurement matrix: matrix W
Original Structure: matrix S
Reconstructed Structure: matrix Shat
Camera Rotations: matrix Rs
# of DCT basis: K
Link list: matrix named list (for drawing sketch figures for mocap datasets)
theta: The camera angular displacement per frame, which was used to create 
W matrix

Real Datasets:
Measurement matrix: matrix W
Reconstructed Structure: matrix Shat
Structure Triangulation: tri
# of DCT basis: K

The dimension of matrix W is 2F-by-P, where F is the total number of
frames and P is total number of points. The image observation for each
image is given in the pair of rows of matrix W. Similar format is used 
for matrix S and Shat. The structure at each time instant is a 3-by-P
matrix and is given in the 3-by-P block of matrix S. Hence the dimension
of S and Shat is 3F-by-P. The dimension of matrix Rs is 2F-by-3, where
each 2-by-3 block represents the camera rotation at that time.

The main function for NRSFM is NRSFM() present in the code. It takes W 
matrix and K (number of basis) as an input and returns the recovered 
structure as output. There are some other optional parameters as well. 
You can easily learn about them by reading the documentation in the 
funtion NRSFM(). Our code also contains a file named example.m, which 
contains some examples on how to call NRSFM().

Some visualization functions are also made part of the code. For each
real dataset seperate visulization function is given. In order to use them,
you might need to see the documentation given in the code. For Mocap datasets
(except danace) you can use viewMocap() function. ViewStruct() function can be
used for all synthetic datasets, but it does not draw sketch figures.

Last thing, please note that NRSFM reconstructs the structure upto a 
Euclidean Transformation. However for synthetic datasets we have aligned S 
and Shat by making both of them zero mean and Procrustes alignment.


In case you have any other confusion or question or suggestion please 
contact me at akhter@lums.edu.pk.

