import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from typing import List, Union
import numpy as np

cmap = ListedColormap([
        '#000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
def plot_task(
    task: dict,
    title: str = None
) -> None:
    """
    displays a task with training examples and test and train inputs
    """
  
    norm = Normalize(vmin=0, vmax=9)
    args = {'cmap': cmap, 'norm': norm}
    
    # Combine train and test examples
    train_examples = task['train']
    test_examples = task['test']
    
    # Calculate total width needed
    train_width = len(train_examples)
    test_width = len(test_examples)
    total_width = train_width + test_width
    
    height = 2  # Always 2 rows (input and output)
    figure_size = (total_width * 3, height * 3)
    figure, axes = plt.subplots(height, total_width, figsize=figure_size)
    
    # Handle single example case
    if total_width == 1:
        axes = axes.reshape(2, 1)
    
    column = 0
    
    # Plot training examples
    for example in train_examples:
        # Get grid dimensions
        input_height, input_width = len(example['input']), len(example['input'][0])
        output_height, output_width = len(example['output']), len(example['output'][0])
        
        # Plot the grids
        axes[0, column].imshow(example['input'], **args)
        axes[1, column].imshow(example['output'], **args)
        
        # Add dimension labels
        axes[0, column].set_title(f'Train Input: {input_height}×{input_width}', fontsize=10)
        axes[1, column].set_title(f'Train Output: {output_height}×{output_width}', fontsize=10)
        
        axes[0, column].axis('off')
        axes[1, column].axis('off')
        column += 1
    
    # Plot test examples
    for example in test_examples:
        # Get grid dimensions
        input_height, input_width = len(example['input']), len(example['input'][0])
                
        # Plot the grids
        axes[0, column].imshow(example['input'], **args)
        
      
        
        # Add dimension labels
        axes[0, column].set_title(f'Test Input: {input_height}×{input_width}', fontsize=10)
        
        axes[0, column].axis('off')
        axes[1, column].axis('off')
        column += 1
    
    if title is not None:
        figure.suptitle(title, fontsize=20)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


def plot_figure(task: List[List[int]], title: str = None) -> None:
    """
    displays a single 2D list (grid)
    """
    height = len(task)
    width = len(task[0]) if height > 0 else 0
    
    # Create figure with appropriate size
    fig_size = (max(width * 0.5, 4), max(height * 0.5, 3))
    figure, axes = plt.subplots(1, 1, figsize=fig_size)
    
    # Plot the grid
    norm = Normalize(vmin=0, vmax=9)
    args = {'cmap': cmap, 'norm': norm}
    axes.imshow(task, **args)
    
    # Add title and labels
    if title is not None:
        axes.set_title(title, fontsize=16)
    axes.set_title(f'Grid: {height}×{width}', fontsize=12)
    
    # Remove axis ticks
    axes.set_xticks([])
    axes.set_yticks([])
    
    plt.tight_layout()
    plt.show()

