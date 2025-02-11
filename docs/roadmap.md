# Roadmap

Below is a high-level roadmap for the CMIP AR7 Fast Track REF  project towards the goal of a public release in October 2025.
This roadmap outlines the key milestones and tasks that need to be completed to achieve this goal.

We have broken down the roadmap into several sections to provide a clear overview of the project's progress.
Namely,

- Metrics
- Infrastructure
- Visualisation
- Testing
- Deployment
- Community Engagement

The roadmap is subject to change as the project progresses, and adjustments may be made based on new insights, feedback, and evolving project requirements.

/// admonition | Note

Click on [] in the top right corner of the diagram to make it full screen.
///

```mermaid
gantt
    dateFormat  YYYY-MM-DD
    title       CMIP REF for AR7 Fast Track
    excludes    weekends

    section Metrics

    Develop list of metrics                       :done, metrics, 2024-10-06, 30d
    Prototype metric package                      :done, metricsPrototype, 2024-10-06, 30d

    Create ESMValTool integration                 :done, esmvaltool, after metricsPrototype, 2025-01-06
    Create ILAMB integration                      :done, ilamb, after metricsPrototype, 2025-01-20
    Create PMP integration                        :active, pmp, after metricsPrototype, 2025-02-16

    Implement missing metrics                     :active, missing, after esmvaltool, 2025-04-01
    Incorporate missing reference datasets        :after beta, 4w
    Fixes                                         :after missing, 6w

    section Infrastructure

    Architecture design                           :done, 2024-10-18, 20d

    Ingest CMIP6 datasets                         :done, cmip6, 2024-11-01, 20d
    Ingest obs4MIPs datasets                      :active, 2025-01-20, 50d
    Develop a local executor                      :done, local, after cmip6, 20d
    Core docker container                         :done, dockerise, 2025-01-10, 10d
    Conda-forge environments                      :conda, after celery, 3w
    Provider docker containers                    :after conda, 3w
    Develop a remote executor                     :active, celery, 2025-01-01, 30d
    Ingest scalar results into database           :results, 2025-03-01, 30d
    Develop a remote executor (slurm)             :slurm, after hackathon, 10d
    Ingest AR7 FT datasets                        :2025-04-20, 4w

    section Visualisation

    CMEC validation                               :done, cmec, 2025-01-15, 20d
    CMEC helpers                                  :active, cmecHelpers, after cmec, 4w
    Basic API for results                         :after celery, 15d
    Integrate Unified Dashboard                   :ud, after cmecHelpers, 20d
    Search results via API                        :api-result, after beta, 2w
    Example prototype dashboard                   :dashboad, after api-result, 6w
    Python package for consuming results          :after api-result, 4w


    section Testing
    Validate package licences                     :done, 2024-11-15, 2024-12-02
    Develop testing framework                     :done, 2024-11-10, 2024-12-10
    Documentation and tutorials                   :active, 2025-03-06, 2025-04-01
    Initial modelling center testing (MetOffice)  :mc1, 2025-03-10, 14d
    Initial modelling center testing (MC 2)       :mc2, after mc1, 14d
    Initial modelling center testing (MC 3)       :mc3, after mc1, 14d
    MB TT stress-testing                          :2025-05-01, 20d
    MB TT Documentation Review                    :2025-05-01, 20d

    section Deployment
    Discuss options with ESGF deployment          :active, esgf, 2024-12-01, 70d
    Build K8 helm charts                          :helm, after esgf, 4w
    ESGF Staging deployment for testing                :staging, after helm, 1w
    CMIP6 stress-test                             :after staging, 4w
    ESGF index integration                        :crit, after beta, 4w
    ESGF Production deployment                         :crit, prod, 2025-08-01, 1w
    Crunch CMIP6 results                          :crit, after prod, 30d

    section Community Engagement
    Model Evaluation Survey                       :done, 2024-05-01, 4w
    Modelling Centre Survey                       :done, 2024-06-01, 4w
    AOGS                                          :done, 2024-06-25, 5d
    Diagnostics Survey                            :done, survey, 2024-10-01, 4w
    ESA CMUG Colocation                           :done, 2024-10-16, 3d
    Project Launch                                :done, 2024-11-04, 1d
    AGU                                           :done, 2024-12-01, 1w
    Hackathon                                     :hackathon, 2025-03-10, 5d
    EGU                                           :egu, 2025-04-20, 1w
    Beta feature freeze                           :milestone, beta, 2025-05-01, 1d
    Beta Release                                  :milestone, beta, 2025-05-30, 1d
    Drop-ins/Engagement                           :engagement, after beta, 2w
    Incorporate feedback                          :after engagement, 6w
    ESA Living Planet Symposium                   :lps, 2025-06-23, 1w
    Public Release                                :milestone, public, 2025-10-30, 1d

```


## Metrics

An overview of the current state of the selected metrics can be found in the
[Metrics Github board](https://github.com/orgs/CMIP-REF/projects/2/views/2) document.
This is being updated as the new metrics are integrated into the CMIP AR7 Fast Track REF.


## What will be in the beta:

We have a beta planned for release at the end of May 2025.
This beta is targeted at allowing the modelling community to test the metrics and provide feedback,
before a public release in October 2025 which will be deployed at ESGF.

The beta will include the following features:

* Ingesting local CMIP6, CMIP AR7 FT, obs4MIPs datasets
    * We will include documention to allow users to ingest their own datasets (we welcome any contributions)
* Examples of ESMValTool, ILAMB, and PMP metrics.
    * Some reference datasets may be missing as we are work through integrating these with obs4MIPs.
* A simple web interface to view the results of any metrics executions locally
* A portrait plot contains results from the 3 different providers
* The ability to run metrics locally, using docker containers (via celery) and via Slurm.
* conda-forge packages for the metrics providers
* Documentation and tutorials

### What is not planned to be in the beta

* Integration with the ESGF indexes - that work is planned for after the beta release
* A publicly available API/website/set of results
  * A private staging deployment is expected by then, but this is for testing purposes only.
* Singularity containers (we welcome any contributions)
* A python API to consume the results easily (results will be available as local files)
* Alerts for when metrics fail
