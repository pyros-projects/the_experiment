from pathlib import Path


def _mk_docker(python_ver):
    fn = Path("Dockerfile")
    if fn.exists():
        return
    packages = Path("packages.txt")
    pkg_line = ""
    reqs = Path("requirements.txt")
    if not reqs.exists():
        reqs.write_text("python-fasthtml\nfasthtml-hf\n")
    req_line = f"RUN pip install --no-cache-dir -r requirements.txt"
    if packages.exists():
        pkglist = " ".join(packages.readlines())
        pkg_line = f"RUN apt-get update -y && apt-get install -y {pkglist}"

    cts = f"""FROM python:{python_ver}
WORKDIR /code
COPY --link --chown=1000 . .
RUN mkdir -p /tmp/cache/
RUN chmod a+rwx -R /tmp/cache/
ENV HF_HUB_CACHE=HF_HOME
{req_line}
{pkg_line}
ENV PYTHONUNBUFFERED=1 PORT=7860
CMD ["python", "main.py"]
"""
    fn.write_text(cts)


_mk_docker("3.12")
