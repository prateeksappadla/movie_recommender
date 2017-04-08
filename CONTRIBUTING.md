CONTRIBUTING.md

Branching Instructions

Clone this repository. Create your own local branch for your code changes. 
Please do not work directly on master branch. Only commit to master branch when you have a working copy of the code that you are working on. This will ensure that the master branch is clean and always has working code. 

git checkout -b mybranch 

When you are ready to commit your changes to master, first rebase your branch onto master branch and then merge your branch with master

git rebase master  /* This may cause merge conflicts, resolve them suitably. Check Note Below for resolving conflicts during rebase */
git checkout master
git merge mybranch

git push -u master origin/master

Now dont forget to switch back to your own branch for further code changes

git checkout mybranch


Note for resolving conflicts during rebase
Open the file that has the conflict
Resolve the conflict
Save the file
Add it to staging area using
git add <filename>

Complete the rebase using
git rebase --continue






