import subprocess


# Run AI_2.py, prediction3.py and plot_1.py in sequence
files_to_run = ['perdiction3.py', 'plot_1.py']

# Prompts the user to choose whether to run api_uden_Date_og_Title.py
run_api_script = input("Do you want to run 'api_uden_Date_og_Title.py'? (y/n): ").strip().lower()

if run_api_script == 'y':
    print(f'Running api_uden_Date_og_Title.py...')
    subprocess.run(['python', 'api_uden_Date_og_Title.py'])



# Run AI_2.py
print(f'Running AI_2.py...')
subprocess.run(['python', 'AI_2.py'])
print(f'AI_2.py completed.\n')


run_ai_2_again=input("Do you want to run 'AI_2.py again?'? (y/n): ").strip().lower()

while run_ai_2_again == 'y':
    print(f'Running AI_2.py again...')
    subprocess.run(['python', 'AI_2.py'])
    print(f'AI_2.py completed.\n')
    run_ai_2_again=input("Do you want to run 'AI_2.py again?'? (y/n): ").strip().lower()



# ask the user to choose running AI_2.py again
#run_api_script = input("Do you want to run 'AI_2.py again?'? (y/n): ").strip().lower()
#if run_api_script == 'y':
#    print(f'Running AI_2.py again...')
#    subprocess.run(['python', 'AI_2.py'])
#    print(f'AI_2.py completed.\n')



# Run the rest of the script
for file in files_to_run:
    print(f'Running {file}...')
    subprocess.run(['python', file])

run_plot_again=input("Do you want to run 'plot_1.py again?'? (y/n): ").strip().lower()

while run_plot_again == 'y':
    print(f'Running plot.py again...')
    subprocess.run(['python', 'plot_1.py'])
    run_plot_again=input("Do you want to run 'plot.py again?'? (y/n): ").strip().lower()

print(f'program completed.\n')
print("you can now close the terminal.")