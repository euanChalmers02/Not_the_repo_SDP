import threading
import queue
import time

# https://www.tutorialspoint.com/python/python_multithreading.htm/
# https://stackoverflow.com/questions/9105990/constantly-looking-for-user-input-in-python

stop = ''

# this will be changes to listening for button presses (with current status vs now status)
def console(q):
    while 1:
        cmd = input('> ')
        q.put(cmd)
        if cmd == 'quit':
            break


def thread_two_action():
    global stop

    run = True
    while run:
        if stop == '':
            print("running thread two")
            time.sleep(2)
        else:
            print("breaking")
            stop = ''
            break

thread2 = threading.Thread(target=thread_two_action)

def pause():
    # how to check if the variable is met throughout the operation??? or is there a better way to kill a thread
    global stop
    # add the voice recording from TextToSpeechHere
    print('--> pause action & kill thread')
    stop = 'stop'


def scanning_mode():
    print('--> running example one')
    thread2.start()


def invalid_input():
    print('---> Unknown command')

if __name__ == '__main__':
    # would add all the button listing here and in the console function
    cmd_actions = {'scanning_mode': scanning_mode, 'p': pause}
    cmd_queue = queue.Queue()

    dj = threading.Thread(target=console, args=(cmd_queue,))
    dj.start()

    while 1:
        cmd = cmd_queue.get()
        if cmd == 'quit':
            break
        action = cmd_actions.get(cmd, invalid_input)
        action()
