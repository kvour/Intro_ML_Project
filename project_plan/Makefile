default:
	latexmk -pdf -pdflatex='pdflatex -shell-escape -file-line-error %O %S -synctex=1 ' root.tex

clean:
	rm -f *.aux
	rm -f *.log
	rm -f *.fdb_latexmk
	rm -f *.bbl
	rm -f *.blg
	rm -f *.dvi
	rm -f *.fls
	rm root.pdf

spell:
	aspell --lang=en --mode=tex check root.tex
