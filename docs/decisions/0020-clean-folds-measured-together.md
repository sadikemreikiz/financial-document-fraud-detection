# 0020. The clean folds are measured once, for all Phase 4 systems together

- **Status**: Accepted
- **Date**: 2026-10-13 (no system has produced an output on the clean folds)

## Context

[ADR-0014](0014-pixel-forensics-preregistration.md) (decision 3) measures each pre-registered
system once on the clean folds: "When development ends, the code is frozen at a commit. Each
system on the pre-registered list of this record is then evaluated on the clean folds once, from
that clean commit." It leaves open whether development ends system by system or for Phase 4 as
a whole. Step 4.3 kept this as its last decision, and
[ADR-0018](0018-global-trace-probe.md) and the
[record of system 1](../system-1-development.md) refer to it.

The systems are at different stages:

- system 1, the JPEG-lattice map ([ADR-0016](0016-system-1-jpeg-trace.md)), is frozen;
- system 5, the global-trace probe, is coded and its development round has run;
- system 6, TruFor ([ADR-0019](0019-system-6-trufor.md)), still needs its development round;
- systems 2 to 4, the learned models, do not exist yet (steps 4.4 and 4.5).

A clean-fold result that is seen before the other systems are designed shapes their design: the
synthetic engine and the learned model could be steered towards what worked for system 1, and
RQ5 compares exactly those systems. A result that has been seen cannot be ignored in practice.

## Options

1. **Each system as soon as it is frozen.** Early numbers, but each result can steer the
   design of the systems still in development.
2. **All systems together, once, after the last is frozen.** No clean result can steer any
   design; the first clean numbers come at the end of Phase 4.
3. **The probe early, the others together.** A shortcut would be known before the learned
   models are trained; but the probe's result would then steer their design, and its
   development round on the design fold already gives that warning without the clean folds (an
   average precision of 0.153 against a chance level of 0.168).

## Decision

1. **Nothing is predicted on the clean folds before every system is frozen.** No score, map,
   metric or per-receipt output of a Phase 4 system is computed on folds 0, 1, 3 or 4 until
   systems 1 to 6 (as amended by ADR-0016 and ADR-0019) are all frozen. Code may still read
   clean-fold receipts to train, as ADR-0014 (decision 1) allows, for example in the development
   rounds that predict fold 2.
2. **One measurement event.** When the last system is frozen, a short record lists the commit at
   which each system was frozen and the clean commit from which all of them run. One command then
   runs every system's clean rounds, writes the run records and generates the report; no result
   of one system is looked at before all have run.
3. **Reruns** only to fix a defect, as ADR-0014 (decision 3) fixes: the defect, the fixing commit
   and both results are reported.

## Consequences

- No clean-fold result can shape a design decision of Phase 4, including the systems that answer
  RQ5.
- Until the end of Phase 4 every number is from the design fold and labelled optimistic; the
  first clean-fold numbers arrive together with the answer to RQ5.
- Freezing becomes an explicit checkpoint: a system's development ends with a record and a
  commit, as system 1's already does.
- A frozen system that later needs a change other than a defect fix becomes a new system, reported
  as such.
