### MWTC 2019 Participant Repository Template
---
This repository is a starting template on top of which you will build your solutions. More importantly, we require that all teams adhere to this directory layout in their code submission.

#### Code Submission
1. We will be using private github repositories for code review and submission this year. Each team is expected to create a [private clone/fork](https://stackoverflow.com/questions/10065526/github-how-to-make-a-fork-of-public-repository-private) of this template repository, and add your solution code without disrupting the directory hierarchy. In particular, we require that ./case3/strategy.py should be directly importable from the root.
2. You should add github account 'uchimwtc' as a collaborator to your private solution repository. In addition, fill in the url for your team private repo in the [spreadsheet](https://docs.google.com/spreadsheets/d/1J-6Ntpoti-S3GeZ_ObabiijltrThdUCsS8HJ7xuB0vs/edit?usp=sharing). We will freeze this spreadsheet from edits on March 22nd, and then invite all teams for a final confirmation.
3. We will pull the latest codebase from your master branch at various deadlines.


#### Package Dependency Management for Case 3
1. Note that only C3 code will be actually run by the case writers. Your solutions for C1 and C2 will be reviewed but not tested on our side. This section is only relevant for C3.
2. We will create a separate [conda environement](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually) for test simulation that is configured by the environment.yml at the root directory. Please update the field 'name' to be your __team name in lower case__ e.g. amherst1. Your canonical team name is the one displayed in the spreadsheet above. We have removed spaces and special symbols.
4. Make sure that your code can run properly in this new environment before the deadline. We will not have the bandwidth to attend to individual dependency management.
3. We reserve the right to disallow certain packages whose computation demands unnecessary resources e.g. GPUs. Everyone knows what they are and let us keep to simple things which would suffice for this case. If you are unsure about whether a particular package is allowed, you should ask on piazza at your earliest convenience. 
