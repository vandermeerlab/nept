vdmlab Core
===========
Objects in vdmlab represent neural and behavioral data.

AnalogSignal
------------
A regular sampling of a continuous, analog signal.

LocalFieldPotential
~~~~~~~~~~~~~~~~~~~
Subclass of AnalogSignal.

Position
~~~~~~~~
Subclass of AnalogSignal, with properties and methods specific 
to 1D and 2D position data.

SpikeTrain
----------
A set of spike times associated with an individual putative neuron.
