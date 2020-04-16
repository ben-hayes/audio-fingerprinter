import curses
import json

statuses = {
    "id_blank_status": {
        "text": """
====================================================================









--------------------------------------------------------------------

====================================================================
            """,
        "y": 0,
        "x": 0
    },
    "id_finished_identifying": {
        "text": """
====================================================================
Finished identifying query: {file_name}
Correctly Identified?       {correctly_identified}
Correct so far:             {correct_so_far}
Guess #1:                   {guess_1}
Guess #2:                   {guess_2}
Guess #3:                   {guess_3}
Time to extract hashes:     {time_to_hashes} seconds
Time to look up in DB:      {time_to_db} seconds
Time elapsed so far:        {total_time} seconds
--------------------------------------------------------------------

====================================================================
            """,
        "y": 0,
        "x": 0
    },
    "id_analysing_file": {
        "text": "Now identifying:            {now_analysing}",
        "y": 12,
        "x": 0
    },
    "id_searching_db": {
        "text": "Searching DB for matches to {now_analysing}...",
        "y": 12,
        "x": 0
    },
    "id_loading_db": {
        "text": "Loading fingerprint database {db_file} from disk...",
        "y": 12,
        "x": 0
    },
    "fp_fingerprint_created": {
        "text": """
====================================================================
Fingerprint created for:    {file_name}
Number of hashes:           {num_hashes}
Number of new hashes:       {num_new_hashes}
Total hashes:               {total_hashes}
Time to create fingerprint: {time_to_create} seconds
Time elapsed so far:        {total_time} seconds
--------------------------------------------------------------------
Now analysing:              
====================================================================
            """,
        "y": 0,
        "x": 0
    },
    "fp_analysing_fingerprint": {
        "text": "Now analysing:              {now_analysing}",
        "y": 9,
        "x": 0
    },
    "fp_writing_db": {
        "text": "Writing fingerprint database {db_file} to disk...",
        "y": 9,
        "x": 0
    }
}

def print_status(status, status_args):
    """
    Helper function for printing the status to the screen.
    
    Arguments:
        screen {curses.window} -- Reference to a curses window object
        status {str} -- Key in statuses dictionary of desired status
        status_args {dict} -- Dict of keyword arguments to format the status
                              string.
    """    
    global statuses
    if print_status.screen is not None:
        # using curses in lieu of print to allow for multiline overwrites
        print_status.screen.addstr(
            statuses[status]["y"],
            statuses[status]["x"],
            statuses[status]["text"].format(**status_args)
        )
        print_status.screen.refresh()


def enable_printing(func):
    """
    A function decorator wrapping it in the curses.wrapper function allowing
    advanced console printing without totally breaking the host terminal.
    
    Arguments:
        func {function} -- Function to be wrapped
    
    Returns:
        function -- The decorated function
    """    
    def intermediate(screen, func, args, kwargs):
        print_status.screen = screen
        curses.use_default_colors()
        func(*args, **kwargs)

    def wrapper(*args, **kwargs):
        curses.wrapper(intermediate, func, args, kwargs)
    
    return wrapper
