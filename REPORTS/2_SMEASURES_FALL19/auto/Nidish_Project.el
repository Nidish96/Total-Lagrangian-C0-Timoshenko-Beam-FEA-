(TeX-add-style-hook
 "Nidish_Project"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "10pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("geometry" "top=1in" "left=.75in" "right=.75in" "bottom=0.75in") ("babel" "USenglish")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "geometry"
    "babel"
    "times"
    "graphicx"
    "subcaption"
    "tikz"
    "hyperref"
    "amsmath"
    "amssymb"
    "cancel"
    "bm"
    "cleveref")
   (LaTeX-add-labels
    "sec:introduction"
    "sec:application-examples"
    "sec:stable-stat-deform"
    "sec:cases-with-bifurc"
    "sec:conclusion"))
 :latex)

