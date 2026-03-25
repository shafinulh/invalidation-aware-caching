import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_nsys_data(db_path, run_name):
    try:
        conn = sqlite3.connect(db_path)
        
        # Determine table names based on typical nsys schema
        # For kernels
        kernel_query = """
        SELECT k.start, k.end, (k.end - k.start) AS duration,
               s.value AS name, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ, k.registersPerThread
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN StringTable s ON k.demangledName = s.id
        """
        try:
            df_kernels = pd.read_sql_query(kernel_query, conn)
        except Exception:
            # Fallback to older nsys schemas or alternative string table
            kernel_query = """
            SELECT k.start, k.end, (k.end - k.start) AS duration,
                   s.value AS name, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ, k.registersPerThread
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            LEFT JOIN StringIds s ON k.demangledName = s.id
            """
            try:
                df_kernels = pd.read_sql_query(kernel_query, conn)
            except Exception as e:
                df_kernels = pd.DataFrame()
                print(f"Warning: Could not extract kernels from {db_path}: {e}")

        # For memory copies
        memcpy_query = """
        SELECT start, end, (end - start) AS duration, bytes, copyKind
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        """
        try:
            df_memcpy = pd.read_sql_query(memcpy_query, conn)
        except Exception as e:
            df_memcpy = pd.DataFrame()
            print(f"Warning: Could not extract memcpy from {db_path}: {e}")

        conn.close()
        
        if not df_kernels.empty:
            df_kernels['run'] = run_name
            df_kernels['duration_ms'] = df_kernels['duration'] / 1e6
        if not df_memcpy.empty:
            df_memcpy['run'] = run_name
            df_memcpy['duration_ms'] = df_memcpy['duration'] / 1e6
            df_memcpy['gb'] = df_memcpy['bytes'] / 1e9
            df_memcpy['gbps'] = df_memcpy['gb'] / (df_memcpy['duration'] / 1e9)
            
        return df_kernels, df_memcpy
    except Exception as e:
        print(f"Error accessing {db_path}: {e}")
        return pd.DataFrame(), pd.DataFrame()

def main():
    os.makedirs('graphs', exist_ok=True)
    
    db_32B = "profiles/nsys_val32B_q_paper_with_plan.sqlite"
    db_1024B = "profiles/nsys_val1024B_q_paper_with_plan.sqlite"
    
    df_k_32, df_m_32 = get_nsys_data(db_32B, "32B")
    df_k_1024, df_m_1024 = get_nsys_data(db_1024B, "1024B")
    
    df_kernels = pd.concat([df_k_32, df_k_1024], ignore_index=True)
    df_memcpy = pd.concat([df_m_32, df_m_1024], ignore_index=True)
    
    if df_kernels.empty or df_memcpy.empty:
        print("Missing data. Cannot generate plots.")
        return

    # Clean kernel names
    df_kernels['short_name'] = df_kernels['name'].astype(str).apply(lambda x: x.split('<')[0].split('(')[0].split('::')[-1] if x else 'Unknown')
    df_kernels['grid_size'] = df_kernels['gridX'] * df_kernels['gridY'] * df_kernels['gridZ']
    df_kernels['block_size'] = df_kernels['blockX'] * df_kernels['blockY'] * df_kernels['blockZ']

    # 1. Kernel Resource & Time Scatter
    plt.figure(figsize=(10, 6))
    
    unique_kernels = df_kernels['short_name'].unique()
    colors = plt.cm.get_cmap('tab10', len(unique_kernels))
    
    for i, kernel in enumerate(unique_kernels):
        mask = df_kernels['short_name'] == kernel
        plt.scatter(
            df_kernels[mask]['grid_size'], 
            df_kernels[mask]['block_size'], 
            s=df_kernels[mask]['duration_ms'] * 5 + 20, 
            label=kernel,
            alpha=0.7,
            color=colors(i)
        )
    
    plt.xscale('log')
    plt.yscale('log', base=2)
    plt.xlabel('Grid Size (Number of Blocks)')
    plt.ylabel('Block Size (Threads per Block)')
    plt.title('Kernel Execution Resources vs Time (Bubble size = Duration)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('graphs/nsight_sqlite_kernel_scatter.png', dpi=300)
    plt.close()

    # 2. Stacked Bar Chart of Execution Breakdown
    # Aggregate top 5 kernels + Memcpy
    top_kernels = df_kernels.groupby('short_name')['duration_ms'].sum().nlargest(4).index
    df_kernels['category'] = df_kernels['short_name'].apply(lambda x: x if x in top_kernels else 'Other Kernels')
    
    # Map copyKind roughly (1=HtoD, 2=DtoH, 8=DtoD typically, or just sum by kind)
    # nsys values: 1: HtoD, 2: DtoH, 3: HtoA, 4: AtoH, 5: AtoA, 6: AtoD, 7: DtoA, 8: DtoD, 9: HtoH, ...
    def map_copy_kind(k):
        if k == 1: return 'Memcpy HtoD'
        if k == 2: return 'Memcpy DtoH'
        if k == 8: return 'Memcpy DtoD'
        return f'Memcpy {k}'
    
    df_memcpy['category'] = df_memcpy['copyKind'].apply(map_copy_kind)
    
    time_k = df_kernels.groupby(['run', 'category'])['duration_ms'].sum().reset_index()
    time_m = df_memcpy.groupby(['run', 'category'])['duration_ms'].sum().reset_index()
    df_time = pd.concat([time_k, time_m], ignore_index=True)
    
    # Pivot for stacked bar
    df_pivot = df_time.pivot(index='run', columns='category', values='duration_ms').fillna(0)
    df_pivot.plot(kind='bar', stacked=True, figsize=(9, 6), colormap='Set3')
    plt.title('Nsight SQLite: Execution Time Breakdown: 32B vs 1024B Values')
    plt.ylabel('Total Time (ms)')
    plt.xlabel('Value Size')
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('graphs/nsight_sqlite_execution_breakdown.png', dpi=300)
    plt.close()

    # 3. I/O Bandwidth & Data Transfer Bar Chart
    plt.figure(figsize=(10, 5))
    df_io = df_memcpy[df_memcpy['category'].isin(['Memcpy HtoD', 'Memcpy DtoH'])]
    io_agg = df_io.groupby(['run', 'category']).agg({'gb': 'sum', 'gbps': 'mean'}).reset_index()
    
    # Simple manual bar plot using matplotlib
    x = range(len(io_agg['run'].unique()))
    runs = sorted(io_agg['run'].unique())
    categories = sorted(io_agg['category'].unique())
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, c in enumerate(categories):
        mask = io_agg['category'] == c
        d = io_agg[mask]
        # Align data with runs
        y = [d[d['run'] == r]['gb'].values[0] if len(d[d['run'] == r]) > 0 else 0 for r in runs]
        bw = [d[d['run'] == r]['gbps'].values[0] if len(d[d['run'] == r]) > 0 else 0 for r in runs]
        
        pos = [pos + (i * width) - (width/2) for pos in x]
        bars = ax.bar(pos, y, width, label=c)
        for j, bar in enumerate(bars):
            if y[j] > 0:
                ax.text(bar.get_x() + bar.get_width()/2., y[j] + 0.001, f'{bw[j]:.1f} GB/s', ha='center', va='bottom')
                
    ax.set_xticks(x)
    ax.set_xticklabels(runs)
    ax.legend()
    plt.title('Total Data Transferred (GB) by Memcpy Type')
    plt.ylabel('Gigabytes (GB)')
    plt.xlabel('Value Size')
            
    plt.tight_layout()
    plt.savefig('graphs/nsight_sqlite_io_bandwidth.png', dpi=300)
    plt.close()

    print("Graphs generated successfully in 'graphs/' directory.")

if __name__ == '__main__':
    main()
