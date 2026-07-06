Added a `register` argument to `ProviderRegistry.build_from_config`.
Registration is the only step that writes to the database, so read-only consumers such as the API can now build the registry with `register=False` and serve a database mounted read-only.
