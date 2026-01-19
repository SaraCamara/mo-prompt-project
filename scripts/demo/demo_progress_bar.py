#!/usr/bin/env python3
"""Demo script to show enhanced progress bar with execution tracker info."""

import time
import sys
from tqdm import tqdm

def format_time(seconds):
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def demo_multi_objective_progress():
    """Simulate multi-objective evolution progress."""
    print("\n" + "="*80)
    print("DEMO: Multi-Objective Evolution Progress Display")
    print("="*80 + "\n")
    
    max_gens = 10
    stagnation_limit = 3
    accumulated_time = 0
    
    with tqdm(range(max_gens), desc="Evolução MOP", unit="gen") as pbar:
        for gen in pbar:
            # Simulate generation work
            gen_duration = 15 + gen * 2  # Increasing time per generation
            time.sleep(0.5)  # Fast demo
            accumulated_time += gen_duration
            
            # Simulate metrics
            best_f1 = 0.65 + gen * 0.03
            pareto_size = 5 + gen % 3
            stagnation = min(gen % 4, stagnation_limit)
            
            # Calculate timing stats
            avg_time = accumulated_time / (gen + 1)
            eta_remaining = avg_time * (max_gens - gen - 1)
            
            # Update progress bar with rich info
            progress_info = {
                "F1": f"{best_f1:.3f}",
                "Pareto": pareto_size,
                "Stag": f"{stagnation}/{stagnation_limit}",
                "Time": format_time(accumulated_time),
                "Avg": format_time(avg_time),
                "ETA": format_time(eta_remaining)
            }
            pbar.set_postfix(progress_info, refresh=True)
    
    print("\n Evolution completed!")
    print(f"   Total time: {format_time(accumulated_time)}")
    print(f"   Final F1: {best_f1:.3f}")


def demo_mono_objective_progress():
    """Simulate mono-objective evolution progress."""
    print("\n" + "="*80)
    print("DEMO: Mono-Objective Evolution Progress Display")
    print("="*80 + "\n")
    
    max_gens = 8
    accumulated_time = 0
    
    with tqdm(range(max_gens), desc="Evolução", unit="gen") as pbar:
        for gen in pbar:
            # Simulate generation work
            gen_duration = 20 + gen * 3
            time.sleep(0.5)  # Fast demo
            accumulated_time += gen_duration
            
            # Simulate metrics
            best_f1 = 0.70 + gen * 0.025
            best_acc = 0.68 + gen * 0.022
            pop_size = 10
            
            # Calculate timing stats
            avg_time = accumulated_time / (gen + 1)
            eta_remaining = avg_time * (max_gens - gen - 1)
            
            # Update progress bar with rich info
            progress_info = {
                "F1": f"{best_f1:.4f}",
                "Acc": f"{best_acc:.3f}",
                "Pop": pop_size,
                "Time": format_time(accumulated_time),
                "Avg": format_time(avg_time),
                "ETA": format_time(eta_remaining)
            }
            pbar.set_postfix(progress_info, refresh=True)
    
    print("\n Evolution completed!")
    print(f"   Total time: {format_time(accumulated_time)}")
    print(f"   Final F1: {best_f1:.4f}")
    print(f"   Final Acc: {best_acc:.3f}")


def demo_resumed_run():
    """Simulate a resumed run with accumulated time."""
    print("\n" + "="*80)
    print("DEMO: Resumed Run (Starting from Generation 5)")
    print("="*80)
    print("Previous session time: 2h 15m")
    print("="*80 + "\n")
    
    max_gens = 10
    start_gen = 5
    previous_time = 2 * 3600 + 15 * 60  # 2h 15m in seconds
    accumulated_time = previous_time
    
    with tqdm(range(start_gen, max_gens), desc="Evolução MOP", unit="gen", 
              initial=start_gen, total=max_gens) as pbar:
        for gen in pbar:
            # Simulate generation work
            gen_duration = 25 + gen * 2
            time.sleep(0.5)  # Fast demo
            accumulated_time += gen_duration
            
            # Simulate metrics
            best_f1 = 0.80 + (gen - start_gen) * 0.015
            pareto_size = 7 + gen % 2
            stagnation = gen % 3
            
            # Calculate timing stats
            avg_time = accumulated_time / (gen + 1)
            eta_remaining = avg_time * (max_gens - gen - 1)
            
            # Update progress bar
            progress_info = {
                "F1": f"{best_f1:.3f}",
                "Pareto": pareto_size,
                "Stag": f"{stagnation}/3",
                "Time": format_time(accumulated_time),
                "Avg": format_time(avg_time),
                "ETA": format_time(eta_remaining)
            }
            pbar.set_postfix(progress_info, refresh=True)
    
    print("\n Evolution completed!")
    print(f"   Total time (including previous session): {format_time(accumulated_time)}")
    print(f"   Current session: {format_time(accumulated_time - previous_time)}")
    print(f"   Final F1: {best_f1:.3f}")


if __name__ == "__main__":
    print("\n" + "█"*80)
    print(" "*20 + "ENHANCED PROGRESS BAR DEMO")
    print("█"*80)
    
    print("\nThis demo shows how the execution tracker information is displayed")
    print("in real-time during evolution runs.\n")
    
    print("Legend:")
    print("  F1       - Best F1 score in current generation")
    print("  Acc      - Best accuracy (mono-objective)")
    print("  Pareto   - Size of Pareto front (multi-objective)")
    print("  Pop      - Population size (mono-objective)")
    print("  Stag     - Stagnation counter / limit")
    print("  Time     - Total accumulated time (across all sessions)")
    print("  Avg      - Average time per generation")
    print("  ETA      - Estimated time remaining")
    
    input("\nPress Enter to start Demo 1: Multi-Objective Evolution...")
    demo_multi_objective_progress()
    
    input("\nPress Enter to start Demo 2: Mono-Objective Evolution...")
    demo_mono_objective_progress()
    
    input("\nPress Enter to start Demo 3: Resumed Run...")
    demo_resumed_run()
    
    print("\n" + "█"*80)
    print(" "*25 + "DEMO COMPLETE!")
    print("█"*80)
    print("\n✨ The same rich information will be displayed during actual evolution runs!")
    print(" All metrics are updated in real-time after each generation.")
    print("  Timing is accurate even across multiple resume sessions.\n")
