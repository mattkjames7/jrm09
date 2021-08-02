# jrm09

JRM09 model (Connerney et al. 2018) implementation using Python.

## Installation

Install using `pip`:

```bash
pip3 install jrm09 --user
```

Or by cloning this repo:

```bash
git clone https://github.com/mattkjames7/jrm09.git
cd jrm09

#EITHER create a wheel and install (replace X.X.X with the version number):
python3 setup.py bdist_wheel
pip3 install dist/jrm09-X.X.X-py3-none-any.whl --user

#OR install directly using setup.py
python3 setup.py install --user
```

## Usage

The model accepts right-handed System III coordinates either in Cartesian form (`jrm09.ModelCart()`) or in spherical polar form (`jrm09.Model()`), e.g.:

```python
import jrm09

#get some Cartesian field vectors (MaxDeg keyword is optional)
Bx,By,Bz = jrm09.ModelCart(x,y,z,MaxDeg=4)

#or spherical polar ones
Br,Bt,Bp = jrm09.Model(r,theta,phi,MaxDeg=4)
```

Please read the docstrings for `jrm09.Model()` and `jrm09.ModelCart()` using `help` or `?` e.g. `help(jrm09.Model)` .

There is also a test function which requires `matplotlib` to be installed:

```python
#evaluate the model at some R
jrm09.Test(R=0.85)
```

which produces this (based on figure 4 of Connerney et al. 2018):

![jrm09test.png](jrm09test.png)

## References

Connerney, J. E. P., Kotsiaros, S., Oliversen, R. J., Espley, J. R.,  Joergensen, J. L., Joergensen, P. S., et al. (2018). A new model of Jupiter's magnetic field from Juno's first nine orbits. Geophysical Research Letters, 45, 2590– 2596. https://doi.org/10.1002/2018GL077312

