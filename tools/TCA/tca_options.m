function options = tca_options(varargin)

%   options = rvm_options('param1',value1,'param2',value2,...)
%
%   Creates an options structure "options" of parameters.
%
%   "options" structure is as follows:
%
%            Field Name:  Description                            : default
% ---------------------------------------------------------------------------------------%
%              'Kernel':  'linear' | 'rbf' | 'poly' | ...        : 'linear'
%         'KernelParam':  --    | sigma | degree                 : 1
%                 'Dim':  no less than zero, integer             : 5
%                  'Mu':  no less than zero                      : 1
%              'lambda':  no less than zero                      : 0
% ---------------------------------------------------------------------------------------%
%
% Acknowledgement: Modified from Vikas Sindhwani's software:
%               http://manifold.cs.uchicago.edu/manifold_regularization/software.html
%
% Author: Sinno Jialin Pan (Dept. of CSE, HKUST)
% June 2008
% ---------------------------------------------------------------------------------------%

% options default values
options = struct('Kernel','linear', ...
    'KernelParam', 1, ...
    'Dim', 5, ...
    'Mu', 1, ...
    'lambda', 0);

numberargs = nargin;

Names = fieldnames(options);
[m,n] = size(Names);
names = lower(Names);

i = 1;
while i <= numberargs
    arg = varargin{i};
    if isstr(arg)
        break;
    end
    if ~isempty(arg)
        if ~isa(arg,'struct')
            error(sprintf('Expected argument %d to be a string parameter name or an options structure.', i));
        end
        for j = 1:m
            if any(strcmp(fieldnames(arg),Names{j,:}))
                val = getfield(arg, Names{j,:});
            else
                val = [];
            end
            if ~isempty(val)
                [valid, errmsg] = checkfield(Names{j,:},val);
                if valid
                    options = setfield(options, Names{j,:},val);
                else
                    error(errmsg);
                end
            end
        end
    end
    i = i + 1;
end

% A finite state machine to parse name-value pairs.
if rem(numberargs-i+1,2) ~= 0
    error('Arguments must occur in name-value pairs.');
end
expectval = 0;
while i <= numberargs
    arg = varargin{i};
    if ~expectval
        if ~isstr(arg)
            error(sprintf('Expected argument %d to be a string parameter name.', i));
        end
        lowArg = lower(arg);
        j = strmatch(lowArg,names);
        if isempty(j)
            error(sprintf('Unrecognized parameter name ''%s''.', arg));
        elseif length(j) > 1
            % Check for any exact matches (in case any names are subsets of others)
            k = strmatch(lowArg,names,'exact');
            if length(k) == 1
                j = k;
            else
                msg = sprintf('Ambiguous parameter name ''%s'' ', arg);
                msg = [msg '(' Names{j(1),:}];
                for k = j(2:length(j))'
                    msg = [msg ', ' Names{k,:}];
                end
                msg = sprintf('%s).', msg);
                error(msg);
            end
        end
        expectval = 1;
    else
        [valid, errmsg] = checkfield(Names{j,:}, arg);
        if valid
            options = setfield(options, Names{j,:}, arg);
        else
            error(errmsg);
        end
        expectval = 0;
    end
    i = i + 1;
end


function [valid, errmsg] = checkfield(field,value)
% CHECKFIELD Check validity of structure field contents.
%   [VALID, MSG] = CHECKFIELD('field',V) checks the contents of the specified
%   value V to be valid for the field 'field'.
%

valid = 1;
errmsg = '';
if isempty(value)
    return
end
isFloat = length(value==1) & isa(value, 'double');
isPositive = isFloat & (value>=0);
isString = isa(value, 'char');
range = [];
requireInt = 0;
switch field
    case 'Kernel'
        if ~isString,
            valid = 0;
            errmsg = sprintf('Invalid value for %s parameter: Must be a string', field);
        elseif ~(strcmp(value, 'linear') | strcmp(value, 'rbf') | strcmp(value, 'rbf_auto') | strcmp(value, 'poly') | strcmp(value, 'invsquare') | strcmp(value, 'lap') | strcmp(value, 'lap_auto') |strcmp(value, 'exp')| strcmp(value, 'inv') )
            valid = 0;
            errmsg = sprintf('Invalid value for %s parameter: Must be in {%s, %s, %s}', field, 'linear', 'rbf', 'rbf_auto', 'lap_auto', 'poly', 'lap', 'invsquare', 'inv', 'exp');
        end
    case 'KernelParam'
        valid = 1;
    case 'Dim'
        valid = 1;
    case 'Mu'
        valid = 1;
    case 'lambda'
        valid = 1;
    otherwise
        valid = 0;
        error('Unknown field name for Options structure.')
end

if ~isempty(range),
    if (value<range(1)) | (value>range(2)),
        valid = 0;
        errmsg = sprintf('Invalid value for %s parameter: Must be scalar in the range [%g..%g]', ...
            field, range(1), range(2));
    end
end

if requireInt & ((value-round(value))~=0),
    valid = 0;
    errmsg = sprintf('Invalid value for %s parameter: Must be integer', ...
        field);
end