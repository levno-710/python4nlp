.PHONY: all clean

REPORT := 3142774_Oelschner_final_project

all: $(REPORT).pdf

$(REPORT).pdf: main.tex references.bib ../results/learning_curve.png
	latexmk -pdf -jobname=$(REPORT) -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -C -jobname=$(REPORT) main.tex
