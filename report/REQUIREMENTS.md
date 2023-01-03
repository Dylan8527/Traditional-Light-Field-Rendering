## Part 1： Traditional Light Field Rendering

### Interpolation

​	Implement two kinds of interpolation schemes to interpolate view along $x$ and $y$ directions.

**WARNING:**

​	Set translation step small enough so as to observe the interpolated views between the data cameras 

**QUESTIONS:**

​	What do you observe if you use biliner interpolation on undersampled light field?

#### bilinear interpolation 

#### qudra-linear interpolation



### Change focal plane.

​	Map the depth of the focal plane to the disparity.

**QUESTIONS:**

- What happens when you move your focal plane from far to near?
- Which focal plane gives yo the optimal results(least aliased reconstruction)?



### Change aperture size.

​	Implement a wide aperture filter to show the effect of changing aperture size.

**HINTS:**

​	You can use Gaussian weight as described in [1].

**QUESTIONS:**

​	What happens when you increase the size of the aperture?



### Expand field of view.

​	Implement the z-directional motion of the camera.



