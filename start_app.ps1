$job1 = Start-Job -ScriptBlock {
    & "the-experiment" --testui 
}
Start-Sleep -Seconds 3
# Start the second program as a background job
$job2 = Start-Job -ScriptBlock {
    & "the-experiment" --app 
}


# Wait for both jobs to complete
Wait-Job $job1, $job2

# Retrieve the output from the jobs
Receive-Job $job1
Receive-Job $job2

# Clean up jobs
Remove-Job $job1, $job2
