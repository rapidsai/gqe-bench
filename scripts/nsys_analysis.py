import sqlite3
import argparse
import csv

def kernel_time(conn, nvtx_range_regex, csv_output, args):
    cursor = conn.cursor()        
    sql_str = f"""
    SELECT 
        nvtx_filt.name,
        COUNT(kernel.correlationId) AS kernel_count,
        ROUND(SUM(kernel.end - kernel.start) / 1000000.0, 3) AS total_kernel_time_ms
    FROM (
        SELECT 
            nvtx.start AS start, nvtx.end AS end, IFNULL(nvtx.text, s.value) AS name
        FROM NVTX_EVENTS nvtx
        LEFT JOIN StringIds AS s ON s.id = nvtx.textId
        WHERE nvtx.eventType = 59 OR nvtx.eventType = 34 OR nvtx.eventType = 75) AS nvtx_filt
    LEFT JOIN CUPTI_ACTIVITY_KIND_KERNEL kernel ON (kernel.start < nvtx_filt.end AND kernel.end > nvtx_filt.start) -- get kernels
    LEFT JOIN StringIds kernel_filter_sid ON (kernel.demangledName = kernel_filter_sid.id) -- get kernel names
    WHERE nvtx_filt.name LIKE "{nvtx_range_regex}" {f"AND kernel_filter_sid.value NOT LIKE \"{args.exclude_kernel_regex}\"" if args.exclude_kernel_regex else ""}
    GROUP BY nvtx_filt.name, nvtx_filt.start, nvtx_filt.end 
    ORDER BY nvtx_filt.start
    """
    print(sql_str)
    cursor.execute(sql_str)
    rows = cursor.fetchall()
    if csv_output:
        with open(csv_output, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Num Kernels", "Total kernel time"])
            writer.writerows(rows)
    else:
        print(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to perform analysis on GQE nsys traces')
    subparsers = parser.add_subparsers(dest="command", required=True)
    parser.add_argument("sqlite", help="Path to sqlite file for analysis")
    parser.add_argument("nvtx_range_regex", help="Regex of NVTX range to analyze")
    parser.add_argument('--csv', default=None, help="CSV file to write to")

    kernel_analysis_parser = subparsers.add_parser("kernel", help="Perform kernel analysis", epilog="Example:\n  python nsys_analysis.py kernel --exclude_kernel_regex \"%fused_concatenate%\" nsys-file.sqlite \"%Run Q13%\"" )
    kernel_analysis_parser.add_argument('--exclude_kernel_regex', default=None, help="Specify regex for kernels which should be excluded from analysis")
    kernel_analysis_parser.set_defaults(program=kernel_time)
    
    args = parser.parse_args()
    conn = sqlite3.connect(args.sqlite)
    args.program(conn, args.nvtx_range_regex, args.csv, args)

