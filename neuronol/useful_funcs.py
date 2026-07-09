from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
from PyPDF2 import PdfMerger
from scipy.stats import shapiro
import textwrap

def mergePDFFiles(src_dir, tar_dir, tar_fname, keywds=['']):
    '''
    Merges all PDF files in `src_dir` with `keywd`(s) present in filenames
    into one PDF file and stores it in `tar_dir` as `tar_fname`.
    '''
    # Find all PDF files containing all keywords
    pdfs = sorted([
        os.path.join(src_dir, f) for f in os.listdir(src_dir)
        if (f[-4:] == '.pdf')
        and np.all([keywd in f for keywd in keywds])
    ])

    # Merge all found PDF files
    fpath_merged = os.path.join(tar_dir, tar_fname)
    with PdfMerger() as merger:
        for pdf in pdfs:
            merger.append(pdf) 
        merger.write(fpath_merged) # save merged PDF file
    
    return None

def test_normality_shapirowilk(data, verbose=False):
    """
    Test for normality of `data` using Shapiro-Wilk Test.
    """
    _, pval = shapiro(data)
    if pval > 0.05:
        if verbose: print("Shapiro-Wilk: Data NORMALLY distributed")
        normality = True
    else:
        if verbose: print("Shapiro-Wilk: Data NOT NORMALLY distributed")
        normality = False
    return normality, pval

def generate_histogram(lst, sort_by=None, ax=None,
                       textwrapping=True):
    """
    Counts the number of occurrences of each unique
    element in `lst` and creates a histogram.
    """
    # Count occurrences
    counts = Counter(lst)

    # Sorting (optional)
    if sort_by is not None and sort_by == 'element':
        counts = dict(sorted(counts.items(), key=lambda x: x[0]))
    elif sort_by is not None and sort_by == 'frequency':
        counts = dict(sorted(counts.items(), key=lambda x: x[1]))
    
    # Get labels and values for the plot
    labels = counts.keys()
    values = counts.values()

    # Wrap long labels (if `lst` contains strings) for readability
    if all(isinstance(w, str) for w in lst):
        if textwrapping:
            labels = [textwrap.fill(w, width=40) for w in labels]

    # Plot histogram (bar chart)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, values)
    ax.set_xlabel("Frequency")

    #   Add labels at the end of the bar
    ax.set_yticks([])
    for i_lbl, (lbl, val) in enumerate(zip(labels, values)):
        ax.text(val + 0.2, i_lbl, lbl,
                va='center', ha='left', fontsize=9)
    return ax