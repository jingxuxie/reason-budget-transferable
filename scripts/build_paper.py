"""Build LaTeX and reject missing references; supports bibtex8 fallback."""
from pathlib import Path
import shutil, subprocess
ROOT=Path(__file__).resolve().parents[1]
pdf=shutil.which('pdflatex')
bib=shutil.which('bibtex') or shutil.which('bibtex8')
if not pdf or not bib:
    raise SystemExit('Install a TeX distribution with pdflatex and bibtex (or bibtex8).')
if not (ROOT/'figures/capping_cost.pdf').exists():
    raise SystemExit('Run make experiments first to regenerate the figure PDFs.')
for command in [[pdf,'-interaction=nonstopmode','-halt-on-error','main.tex'],[bib,'main']]+[[pdf,'-interaction=nonstopmode','-halt-on-error','main.tex']]*3:
    result=subprocess.run(command,cwd=ROOT/'paper',text=True,capture_output=True)
    if result.returncode:
        print(result.stdout);print(result.stderr);raise SystemExit(result.returncode)
log=(ROOT/'paper/main.log').read_text(errors='replace')
for token in ['There were undefined','Overfull \\hbox','Overfull \\vbox']:
    if token in log:raise SystemExit('Inspect LaTeX log: '+token)
print('Built paper/main.pdf with resolved references and no overfull boxes.')
