# License Plate Number Detection

## Article Summary

This article explores the foundational stage of License Plate Recognition (LPR) systems, focusing specifically on License Plate Number Detection (LPND)—the process of identifying the location and boundaries of a vehicle’s license plate within an image. It introduces several classic detection techniques, such as object contour analysis, segmented boundary analysis, and histogram-based methods, outlining their strengths, limitations, and use cases.

Furthermore, the article emphasizes the critical role of hardware quality (e.g., camera resolution and frame clarity) in ensuring reliable detection, before delving into a practical Python implementation. Although Optical Character Recognition (OCR) is mentioned as a later step in the LPR pipeline, the focus remains strictly on the detection phase.

By the end, readers gain both theoretical knowledge and hands-on tools to begin experimenting with LPND themselves.



## Getting Started

For this project, we will be using Python 3.11.4. Follow these steps to get started:

1. **Install Requirements**: Install the required packages from the `requirements.txt` file using the following command:

   ```sh
   pip install -r requirements.txt
   ```

2. **Run the executable file**: Run the `main1.py` and `main2.py` files in your code editor to see the examples in action.

## Source Code Explanation

The code provided in this project demonstrates basic examples of traditional techniques used for **License Plate Number Detection (LPND)**. Here's a breakdown of the key components:

- `src/main1.py` and `src/main2.py`: These scripts contain test implementations of LPND methods discussed in the article.

- `img/testing/`: This directory holds the test images used to evaluate the detection algorithms.

