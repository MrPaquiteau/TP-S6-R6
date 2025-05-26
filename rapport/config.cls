\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tpreport}[2024/01/01 TP Report Class]

% Load base class
\LoadClass{article}

% Required packages
\RequirePackage{graphicx}
\RequirePackage{float}
\RequirePackage[colorlinks=true, urlcolor=blue, linkcolor=black, citecolor=black]{hyperref}
\RequirePackage[margin=1in]{geometry}
\RequirePackage[french]{babel}
\RequirePackage{datetime}

% Custom commands and settings
\renewcommand{\contentsname}{Table des matières}

% Title page formatting
\renewcommand{\maketitle}{%
    \begin{titlepage}
        \centering
        \vspace*{2cm}
        {\Huge\bfseries\@title\par}
        \vspace{2cm}
        {\Large\@author\par}
        \vspace{1cm}
        {\large\@date\par}
        \vfill
    \end{titlepage}
}