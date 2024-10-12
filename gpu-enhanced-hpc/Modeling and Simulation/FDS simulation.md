To get started with an FDS [[simulation]], you should try using an existing simple input file. A recommended example for beginners is the `simple_test.fds` file located in the subfolder called `Fires` within the `Examples` folder of the standard FDS installation. This particular case is designed as a simple single mesh model and running it should help you become familiar with the process. You can find this file and run it to verify your installation and understand the basics of [[simulation]] and output analysis using Smokeview`[1]`.

To verify that an FDS [[simulation]] ran properly, you can follow these steps:

1. **Check the Output Files**:
    
    - When you run an FDS [[simulation]], it generates several output files. Look for `.out`, `.log`, and `.smv` files.
    - The `.out` file is a text file that contains information about the progress and completion of the [[simulation]]. It should indicate that the [[simulation]] finished without errors.
    - The `.log` file can contain warnings or errors that occurred during the run.
2. **Review the [[simulation]] Time**:
    
    - Ensure the [[simulation]] runs for the expected duration. The [[simulation]] time is defined in your `.fds` input file under the `&TIME` section.
3. **Examine the Console Output**:
    
    - Check the console output for any errors or warnings during the [[simulation]]. FDS usually reports issues if they occur.
4. **Visualize with Smokeview**:
    
    - Use Smokeview to visualize the results by opening the `.smv` file. This visualization tool will help you check if the [[simulation]] behaves as anticipated.
    - Look for expected fire behavior or smoke movement, ensuring that it matches the specifications of your input file.
5. **Check for Numerical Instability**:
    
    - Make sure there aren't any obvious numerical instabilities or negative temperatures which would indicate something went wrong in the model setup or during the [[simulation]].
6. **Cross-Reference with Documentation**:
    
    - Compare your output with expected results found in the FDS documentation or examples. If you are using a known example, it may have a reference output you can compare against.

By following these steps, you should be able to confirm that your FDS run was successful and determine if there are any issues that need addressing.