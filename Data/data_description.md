Data Description
Overview
This folder contains home non-invasive ventilator recordings from 87 COPD patients.
Data is collected passively during nightly ventilator use at 5 Hz (one reading every 0.2 seconds).
Patient Split
SplitPatientsLabel 0Label 1Train503119Val15105Test22175Total875730

Label 0 — no acute exacerbation during monitoring period
Label 1 — severe AECOPD requiring ICU-level emergency admission

Columns
ColumnDescriptionUnittimestampDate and time of readingdatetimeflowInspiratory/expiratory flow rateL/minpressureCircuit air pressurecmH₂OSpO2Peripheral oxygen saturation%ResRateRespiratory ratebreaths/minTidalVolumeVolume exhaled per breathmLMinuteVentTotal ventilation per minuteL/minLeakSystem air leakageL/minlabel0 = healthy, 1 = severe AECOPDbinarypatient_idAnonymised patient identifierinteger
