# Rendering test — markdown

Sanity check that VSCode's markdown preview on SCC handles the `img/` paths
correctly. If these images render in the preview pane, we can use the same
pattern for the qualitative-harness output reports.

## Per-direction figures (English → Spanish)

### Mean-abs IE heatmap (heads)
![English→Spanish heads mean-abs IE](img/English__Spanish_heads_heatmap.png)

### Signed mean IE heatmap (heads)
![English→Spanish heads signed IE](img/English__Spanish_heads_signed_heatmap.png)

### Top-20 heads bar
![English→Spanish top heads](img/English__Spanish_top_heads_bar.png)

### Top-50 SAE features bar
![English→Spanish top SAE features](img/English__Spanish_top_sae_bar.png)

## Cross-direction figure

### Universal heads heatmap (directions-in-top-K per (layer, head))
![Universal heads](img/universal_heads_heatmap.png)
