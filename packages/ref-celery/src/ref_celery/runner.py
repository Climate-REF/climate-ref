from ref_celery.app import create_celery_app


def main():
    """
    Send a task to the workers
    """
    app = create_celery_app("ref_celery")

    # Inquire what tasks are available
    i = app.control.inspect()
    print(i.registered())

    res = app.send_task("example_example")

    print(res.get(timeout=10))


if __name__ == "__main__":
    main()
