<!-- terrarium-managed:start -->
<!-- terrarium-version: 0.1.0 -->
# Terrarium Rust Tools

This project uses the `rust` terrarium preset. During `terrarium run`, use the built-in terrarium MCP tools for Rust workflow commands instead of direct shell commands when possible.

Prefer the terrarium MCP tools over direct shell commands for normal Rust build workflows:

- `cargo_check`
- `cargo_test`
- `cargo_build`
- `cargo_clippy`
- `cargo_fmt_check`
- `cargo_fmt`
- `cargo_update`
- `cargo_upgrade_incompatible`
- `make_install` (only when Makefile exists)
- `make_run`
- `docker_run`
- `command_log_read`
- `spenn_list_scenarios`
- `spenn_run_scenario`
- `git_status`
- `git_unmerged`
- `git_log`
- `git_diff`
- `git_show`
- `report_violation`

You are running inside a macOS sandbox with a deny-default policy. If you encounter a permission denied error or an operation blocked by the sandbox, call `report_violation` with a description of what was attempted and the error.

Use the built-in git MCP tools for read-only repository inspection instead of direct shell `git` commands.
For linked repositories, use the git tool `path` parameter to select the repository; use `pathspec` only for files inside that repository.
When implementing a Rust feature, keep `Cargo.toml` aligned with the work. If dependencies or crate features changed, update `Cargo.toml` and refresh `Cargo.lock` with `cargo_update`.
`cargo_test` can run the full suite, a single test, or a filtered subset.
After implementing a Rust feature, run the tests with `cargo_test`.
Never modify tests to make them pass — tests are the source of truth. Fix the implementation to satisfy the tests. Only change a test if it does not compile. If you believe a test's intention is wrong, ask the user before changing it.
When resolving merge conflicts, diff against the source branch and verify the intention of the code is preserved. Ask the user if anything is unclear.
Always fix all compiler warnings and clippy lints — treat warnings as errors.
<!-- terrarium-managed:end -->

