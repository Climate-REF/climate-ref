# CHANGELOG

This directory contains "news fragments", i.e. short files that contain a small markdown-formatted bit of text that will be
added to the CHANGELOG when it is next compiled.

The CHANGELOG will be read by users,
so this description should be aimed to CMIP Rapid Evaluation Framework users
instead of describing internal changes which are only relevant to developers.
Merge requests in combination with our git history provides additional
developer-centric information.

Make sure to use phrases in the past tense and use punctuation, examples:

```
Improved verbose diff output with sequences.

Terminal summary statistics now use multiple colors.
```

Each file should have a name of the form `<PR>.<TYPE>.md`, where `<PR>` is the pull request number, and `<TYPE>` is one of:

* `highlights`: a short summary of the release, written once per release.
* `feature`: new user facing features, like new command-line options and new behaviour.
* `improvement`: improvement of existing functionality, usually without requiring user intervention
* `fix`: fixes a bug.
* `docs`: documentation improvement, like rewording an entire section or adding missing docs.
* `deprecation`: feature deprecation.
* `breaking`: a change which may break existing uses, such as feature removal or behaviour change.
* `trivial`: fixing a small typo or internal change that might be noteworthy.

So for example: `123.feature.md`, `456.fix.md`.

Since you need the pull request number for the filename, you must submit a pull request first.
From this pull request, you can get the pull request number and then create the news file.
A single pull request can also have multiple news items,
for example a given pull request may add a feature as well as deprecate some existing functionality.

If you are not sure what issue type to use, don't hesitate to ask in your pull request.

`towncrier` preserves multiple paragraphs and formatting (code blocks, lists, and so on),
but for entries other than features
it is usually better to stick to a single paragraph to keep it concise.
TODO: Update the below to reflect the new approach to cross-referencing entries.
You may also use [markdown style cross-referencing](https://mkdocstrings.github.io/usage/#cross-references) within your news items to link to other
documentation.

You can also run `towncrier build --draft` to see the draft changelog that will be appended to [docs/changelog.md]()
on the next release.
