from services.api.app.core.config import load_settings


def main() -> None:
    settings = load_settings()
    import uvicorn

    uvicorn.run(
        "services.api.app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
