#!/usr/bin/env perl

$latex                       = 'platex -synctex=1 -halt-on-error %O %S';
$bibtex                      = 'pbibtex %O %B';
$dvipdf                      = 'dvipdfmx %O -o %D %S';
$makeindex                   = 'mendex -U %O -o %D %S';
$max_repeat                  = 5;
$pdf_mode                    = 3; # generates pdf via dvipdfmx

# Prevent latexmk from removing PDF after typeset.
# This enables Skim to chase the update in PDF automatically.
$pvc_view_file_via_temporary = 0;

## PDF Previewr
# Acrobat Reader を使う場合 (非推奨。xdvi / xpdf を利用するのが良い)
#$pdf_previewer = 'open -a Adobe\ Reader.app %S';

# Use Skim as a previewer
#$pdf_previewer               = 'open -a /Applications/Skim.app'; # Skimの場所を指定する
$pdf_previewer = 'evince';
