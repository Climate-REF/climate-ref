ESMValTool series now emit the first-class `kind` (`model` or `reference`),
so an observation curve can be told apart from a model curve directly
instead of being inferred from the presence of `reference_source_id`.
The presentation attributes carried by each series are also made uniform across diagnostics.
