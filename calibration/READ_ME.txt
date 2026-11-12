This calibration data comprises a set of DWI scans collected on the Hyperfine MRI scanner at the Clinical Research
Facility, King's College Hospital, London.

The dataset comprises scans for every combination of the following sequence parameters:

    b = [0, 200, 400, 600, 800]
    dwi direction = [X, Y, Z]
    dwi gradient duration (delta) = 30, 35, 40 ms

The data was collected in a nested set of loops:

    - Outer loop:   gradient duration
    - Middle loop:  axis
    - Inner loop:   b-value

At times the protocol would fail and then be started again and hence the full dataset was collected across multiple days
and there are slightly differing numbers of repeats for each parameter combination. Be careful to account for this when processing
the dataset to produce calibration maps as otherwise this will bias the calibration method to certain sequence parameter combinations.

Note, temperature measurements were collected throughout however at one point it appears the temperature sensor (optical thermometer gun)
ran out of power and turned off, hence there are certain scans/sequences that do not have corresponding temperature measurements.
These scans therefore should be removed from the dataset in pre-processing.