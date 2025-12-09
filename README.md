# sensible-fitting
An overarching framework wrapping fitting libraries/functions providing a common API for different backends. It (opinionatedly) focuses on fit functions related to AMO physics. It helps to guess seed parameters and provides a unified and sensible API that is quick to use. This library is currently just used in our labs, but I indend to make it 'nice' for public good at some point.

This project was born out of the desire to replace the (`oitg`)[https://github.com/OxfordIonTrapGroup/oitg]
dependency of (`ndscan`)[https://github.com/OxfordIonTrapGroup/ndscan]. This dependency primarily comes through the plotting functions.
