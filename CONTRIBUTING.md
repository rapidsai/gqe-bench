# Contributing to GQE Bench

Contributions to GQE Bench fall into the following categories:

1. To report a bug, request a new feature, or report a problem with documentation, please file an
   [issue](https://github.com/rapidsai/gqe-bench/issues/new/choose) describing the problem or new feature in detail. The GQE team
   evaluates and triages issues, and schedules them for a release. If you believe the issue needs
   priority attention, please comment on the issue to notify the team.
2. To propose and implement a new feature, please file a new feature request
   [issue](https://github.com/rapidsai/gqe-bench/issues/new/choose). Describe the intended feature and
   discuss the design and implementation with the team and community. Once the team agrees that the
   plan looks good, go ahead and implement it, using the [code contributions](#code-contributions)
   guide below.
3. To implement a feature or bug fix for an existing issue, please follow the [code
   contributions](#code-contributions) guide below. If you need more context on a particular issue,
   please ask in a comment.

## Code contributions

### Initiating contribution

Currently, GQE Bench development is internally driven. Public contributions to GQE Bench should be initiated as exploratory discussions with the team to evaluate the proposed improvement or fix. Should public code contributions be deemed advisable through these discussions, the following guidelines are recommended.

### Your first issue

1. Follow the guide at the bottom of this page for
   [Setting up your GQE Bench environment](#setting-up-your-gqe-bench-environment).
2. Find an issue to work on. The best way is to look for the
   [good first issue](https://github.com/rapidsai/gqe-bench/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
   or [help wanted](https://github.com/rapidsai/gqe-bench/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
   labels.
3. Comment on the issue stating that you are going to work on it.
4. Create a fork of the GQE Bench repository and check out a branch with a name that
   describes your planned work. For example, `fix-documentation`.
5. Write code to address the issue or implement the feature.
6. Add unit tests and unit benchmarks.
7. [Create your pull request](https://github.com/rapidsai/gqe-bench/compare). To run continuous integration (CI) tests without requesting review, open a draft pull request.
8. Verify that CI passes all [status checks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/about-status-checks).
   Fix if needed.
9. Wait for other developers to review your code and update code as needed.
   Changes require approval from GQE Bench maintainers before merging.
10. Once reviewed and approved, a GQE Bench developer will merge your pull request.

If you are unsure about anything, don't hesitate to comment on issues and ask for clarification!

### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you can look at the
prioritized issues for our next release in our
[project boards](https://github.com/rapidsai/gqe-bench/projects).

Look at the unassigned issues, and find an issue to which you are comfortable contributing. Start
with _Step 3_ above, commenting on the issue to let others know you are working on it. If you have
any questions related to the implementation of the issue, ask them in the issue instead of the PR.

## Setting up your GQE Bench environment
1. Begin by cloning the GQE Bench repository in your workspace:
   ```bash
   git clone <GQE_BENCH_REPO>
   ```
2. The easiest way to set up the GQE Bench environment is by running the appropirate GQE [Docker image](https://github.com/orgs/rapidsai/packages?repo_name=gqe) which has all dependencies installed. Follow the [Docker Engine install instructions](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) if you need to install Docker. If needed, [install](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt) and [configure](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker) the NVIDIA container toolkit. With Docker set up, launch the GQE container:
   ```bash
   docker run -it --rm --gpus all \
            -v gqe-bench:<GQE_BENCH_DIR> \
            -v <DATASET_DIR>:<e.g. /sf100_id32> \
            -v <REFERENCE_RESULTS_DIR>:<e.g. /solution> \
            -v <QUERY_PLAN_DIR>:<e.g. /plan> \
            -v <HOST_DIR>:<CONTAINER_DIR> \
            <GQE_DOCKER_IMAGE>
   ```
3. Check that the GQE conda environment is activated. If not, activate it with the following:
   ```bash
   conda activate gqe
   ```
4. Now you are all set to install and run GQE Bench by following the steps in [README.md](README.md).

## Developer Guidelines

### Python coding style
We generally follow [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/).

### C++ coding style
For high-level design issues like interfaces, class hierarchies, recourse management, error handling etc., we will follow [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines). As an auxiliary repo to GQE, we will also follow [GQE developer guide](https://github.com/rapidsai/gqe/tree/main?tab=contributing-ov-file#developer-guidelines) where relevant.

### Code formatting
GQE Bench uses [pre-commit](https://pre-commit.com/) to enforce code style. Please see details in the [relevant section of README](README.md#pre-commit-hooks).

## Signing Your Work

We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```
