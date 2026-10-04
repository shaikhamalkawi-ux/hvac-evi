# Source and claim boundaries

## Evidence sources

### ORNL
Measured controlled test-building experiments. Used to study complete-day dependence and related sensitivity analyses. This is not treated as occupied-building operational validation.

### LBNL boiler
One simulated plant. Used for matched calendar evaluation, sensor-tier analysis, held-intensity checks, and resolution sensitivity. Simulation is not described as field validation.

### PreDist v1
Operational district-heating substation time series linked to service reports. Used for complete-substation evaluation and target-support diagnostics. Service reports do not establish exact physical fault onset.

### Wang multi-building AHUs
Published real-operational BMS data from 41 AHUs:
- 13 auditorium AHUs;
- 8 hospital AHUs;
- 20 office AHUs.

The labels are expert/criteria-based operational annotations. They are not upgraded to repair-confirmed physical-failure truth.

## External-validation admission

The Wang source is admitted for:
- within-building equipment-independent evaluation;
- six ordered cross-building score comparisons;
- source-defined target-support diagnostics.

It is not treated as:
- an author-run field experiment;
- repair-confirmed fault validation;
- an energy-savings trial;
- an operator-action or maintenance-outcome trial.

## Primary external-analysis universe

Classes:
- Normal
- RATSF
- SFF

Common sensor semantics:
1. set-point temperature
2. return-air temperature
3. supply-air temperature
4. supply fan
5. valve position
6. heating-supply temperature
7. cooling-supply temperature
8. cooling pump

Heating-pump semantics are excluded from the common intersection because the exact hospital source does not provide a supported equivalent.

## Reporting boundary

A favourable aggregate score does not by itself establish:
- equipment-independent generalisation;
- preserved weakest-class protection;
- cross-building transportability;
- deployment readiness.
