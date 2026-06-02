import matplotlib.pyplot as plt



def plot_avg_placements(competition, average_placement_table, group):

    pct_fin_group = average_placement_table.sort_values(1, axis=1, ascending=False)

    fig, ax = plt.subplots(figsize=(20,10), facecolor='white')

    ax.axis('off')
    ax.axis('tight')
    ax.set_title(f"Group {group}", fontsize=18, fontweight='bold')

    table = ax.table(cellText=pct_fin_group.values, 
                        cellColours=plt.cm.YlOrRd(pct_fin_group.values),
                        rowLabels=pct_fin_group.index, colLabels=pct_fin_group.columns, 
                        loc='center'
                        )

    table.scale(1,1.7)

    table.auto_set_font_size(False)
    table.set_fontsize(14)

    for val in range(1,5):
        row = table[val,-1]
        row.set_text_props(fontsize=14, fontweight='bold', verticalalignment='center')
        row.PAD = 0.4

        head = table[0,val-1]
        head.set_text_props(fontsize=14, fontweight='bold', verticalalignment='center')

    plt.savefig(f'./TM_Outputs/{competition}/{group_placement_table}')